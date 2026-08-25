"""
AI Text to Organized Multi-Language PDF Generator -- Backend

Pipeline:

    raw text
        -> unicode NFC normalization (safe for Devanagari; no other cleanup)
        -> split into ordered (text, directive) segments
             directives ([[image:N]], ---, ***, ___) NEVER go through
             Gemini or the fallback parser's text logic -- they become
             blocks directly, deterministically, every time. This closes
             a real gap: Gemini's JSON schema has no slot for dividers or
             images, so routing them through Gemini would silently drop
             them.
        -> each text segment is hierarchically chunked for Gemini sizing
           (paragraph -> sentence -> word boundary -> hard cut, never
           mid-word), only the very first chunk of the very first text
           segment in the whole document is ever title-eligible
        -> [optional, per chunk] Gemini structural classification
        -> exact token-level lossless verification (see gemini_client.py)
        -> deterministic fallback parser for any chunk that Gemini
           couldn't handle or that failed verification (allow_title
           passed through explicitly -- this is the fix for the
           mid-document fallback title-loss bug)
        -> validated Document model (engine-independent)
        -> HTML generation (structure) + CSS (theme, generated in Python)
        -> WeasyPrint PDF rendering (Pango/HarfBuzz text shaping)
        -> final PDF

Run:
    pip install -r requirements.txt
    python app.py

Production:
    gunicorn app:app --workers 2 --threads 4 --timeout 180 --bind 0.0.0.0:$PORT
"""

import os
import re
import io
import time
import logging
import unicodedata
from typing import List, Optional, Tuple

from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS

import gemini_client
import renderer
from document_model import Document, Block
from fallback_parser import parse_fallback

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("app")

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 60 * 1024 * 1024  # 60 MB request cap

DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() == "true"
MAX_TEXT_CHARS = 3_000_000
MAX_IMAGES = 40

_last_debug = {
    "raw": None,
    "normalized": None,
    "segments": None,
    "document_json": None,
    "html": None,
    "timings": None,
    "chunk_results": None,
}

renderer.ensure_fonts_and_engine()

_DIRECTIVE_LINE_RE = re.compile(r'^(\[\[image:[A-Za-z0-9_-]+\]\]|---|\*\*\*|___)$')
_IMAGE_MARKER_RE = re.compile(r'^\[\[image:([A-Za-z0-9_-]+)\]\]$')


def api_error(code: str, message: str, http_status: int = 400):
    return jsonify({"error": {"code": code, "message": message}}), http_status


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------
def normalize_text(raw_text: str) -> str:
    """
    Unicode NFC normalization ONLY. NFC composes canonical forms without
    reordering or altering Devanagari matras/conjuncts -- it is the
    correct, safe normalization form for mixed-script text. Deliberately
    does NOT trim, collapse whitespace, or strip anything else here --
    any such change would risk silently altering content, which is the
    top priority to avoid. (Whitespace normalization is used ELSEWHERE,
    but only transiently, for validation comparison -- never applied to
    text that gets stored or rendered.)
    """
    return unicodedata.normalize('NFC', raw_text.replace('\r\n', '\n').replace('\r', '\n'))


# ---------------------------------------------------------------------------
# Segment splitting: directives vs. text runs
# ---------------------------------------------------------------------------
def split_into_segments(text: str) -> List[Tuple[str, str]]:
    """
    Split into an ordered list of ("text", content) / ("directive", content)
    tuples. A directive is a line that is EXACTLY an image marker or a
    divider -- these are structural instructions, not prose, and must
    never be interpreted or altered by Gemini or the prose-oriented parts
    of the fallback parser. Every other line is accumulated into "text"
    runs, preserving blank lines (paragraph boundaries) within each run.
    """
    lines = text.split('\n')
    segments: List[Tuple[str, str]] = []
    buf: List[str] = []

    for line in lines:
        stripped = line.strip()
        if _DIRECTIVE_LINE_RE.match(stripped):
            if buf:
                segments.append(("text", '\n'.join(buf)))
                buf = []
            segments.append(("directive", stripped))
        else:
            buf.append(line)

    if buf:
        segments.append(("text", '\n'.join(buf)))

    return segments


def directive_to_block(content: str) -> Optional[Block]:
    if content in ('---', '***', '___'):
        return Block(type="divider")
    m = _IMAGE_MARKER_RE.match(content)
    if m:
        return Block(type="image", image_index=m.group(1))
    return None


# ---------------------------------------------------------------------------
# Document building
# ---------------------------------------------------------------------------
def build_document(normalized_text: str, use_ai: bool) -> Tuple[Document, float, List[dict]]:
    """
    Core structuring step. Returns (document, gemini_analysis_ms_total,
    chunk_results) where chunk_results is per-chunk debug/audit info.

    Title eligibility: only the very first chunk of the very first TEXT
    segment in the entire document may become the document title -- every
    other chunk, from any segment, is processed with allow_title=False,
    whether it goes through Gemini or the fallback parser. This is what
    prevents a fallback chunk in the middle of the document from stealing
    the title slot and losing its own first line.
    """
    segments = split_into_segments(normalized_text)

    all_blocks: List[Block] = []
    title = ""
    any_gemini_used = False
    any_fallback_used = False
    title_eligible = True
    gemini_analysis_ms_total = 0.0
    chunk_results: List[dict] = []

    ai_usable = use_ai and gemini_client.is_configured()

    for seg_type, content in segments:
        if seg_type == "directive":
            block = directive_to_block(content)
            if block:
                all_blocks.append(block)
            continue

        if not content.strip():
            continue

        sub_chunks = gemini_client.chunk_for_gemini(content) if ai_usable else [content]

        for idx, sub_chunk in enumerate(sub_chunks):
            allow_title = title_eligible and idx == 0

            if ai_usable:
                parsed_json, validated_blocks, ok, analysis_ms, error_info = gemini_client.structure_chunk(
                    sub_chunk, allow_title=allow_title
                )
                gemini_analysis_ms_total += analysis_ms
                chunk_results.append({
                    "chars": len(sub_chunk),
                    "allow_title": allow_title,
                    "gemini_ok": ok,
                    "analysis_ms": analysis_ms,
                    "error": error_info,
                })

                if ok and validated_blocks is not None:
                    any_gemini_used = True
                    if allow_title and parsed_json and parsed_json.get("title"):
                        title = parsed_json["title"]
                    all_blocks.extend(validated_blocks)
                    if allow_title:
                        title_eligible = False
                    continue

                # Gemini failed, invalid, or failed lossless verification --
                # fall back for THIS sub-chunk only. The rest of the
                # document's already-accepted Gemini results are untouched.
                any_fallback_used = True

            fb_doc = parse_fallback(sub_chunk, allow_title=allow_title)
            if allow_title and fb_doc.title:
                title = fb_doc.title
            all_blocks.extend(fb_doc.blocks)
            if allow_title:
                title_eligible = False

    doc = Document(title=title, blocks=all_blocks)
    if any_gemini_used and any_fallback_used:
        doc.source = "mixed"
        doc.warnings.append("Some sections used basic formatting because AI structuring could not be verified for them.")
    elif any_gemini_used:
        doc.source = "gemini"
    elif use_ai and not gemini_client.is_configured():
        doc.source = "fallback"
        doc.warnings.append("AI structuring unavailable — using basic formatting.")
    else:
        doc.source = "fallback"

    return doc, gemini_analysis_ms_total, chunk_results


def document_to_json(doc: Document) -> dict:
    return {
        "title": doc.title,
        "source": doc.source,
        "warnings": doc.warnings,
        "blocks": [
            {
                "type": b.type,
                "text": b.text,
                "level": b.level,
                "items": b.items,
                "rows": b.rows,
                "image_index": b.image_index,
            }
            for b in doc.blocks
        ],
    }


# ---------------------------------------------------------------------------
# Full pipeline (structuring + render), with separated timing
# ---------------------------------------------------------------------------
def run_pipeline(raw_text: str, use_ai: bool, images_map: dict, theme: dict, title_override: Optional[str]):
    timings = {}

    t0 = time.perf_counter()
    normalized = normalize_text(raw_text)
    timings["normalize_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    t1 = time.perf_counter()
    document, analysis_ms, chunk_results = build_document(normalized, use_ai)
    if title_override:
        document.title = title_override
    structuring_ms = round((time.perf_counter() - t1) * 1000, 1)
    timings["analysis_ms"] = round(analysis_ms, 1)          # Gemini-only time
    timings["structuring_ms"] = structuring_ms              # analysis + fallback + validation
    timings["structuring_method"] = document.source

    t2 = time.perf_counter()
    html_string = renderer.build_html(document, images_map, theme)
    timings["html_ms"] = round((time.perf_counter() - t2) * 1000, 1)

    t3 = time.perf_counter()
    pdf_bytes = renderer.render_pdf(html_string)
    timings["render_ms"] = round((time.perf_counter() - t3) * 1000, 1)

    timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    if DEBUG_MODE:
        _last_debug["raw"] = raw_text
        _last_debug["normalized"] = normalized
        _last_debug["segments"] = split_into_segments(normalized)
        _last_debug["document_json"] = document_to_json(document)
        _last_debug["html"] = html_string
        _last_debug["timings"] = timings
        _last_debug["chunk_results"] = chunk_results

    return pdf_bytes, document, timings


# ---------------------------------------------------------------------------
# Routes: health & status
# ---------------------------------------------------------------------------
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "pdf-backend"})


@app.route('/api/status', methods=['GET'])
def status():
    force = request.args.get('force', 'false').lower() == 'true'
    gemini_result = gemini_client.check_health(force=force)
    return jsonify({
        "backend": {
            "status": "online",
            "fonts_ready": renderer.fonts_ready(),
        },
        "gemini": {
            "status": gemini_result["status"],
            "reason": gemini_result.get("reason"),
            "response_time_ms": gemini_result.get("response_time_ms"),
            "checked_at": gemini_result.get("checked_at"),
            "model": gemini_client.GEMINI_MODEL,
        },
    })


# ---------------------------------------------------------------------------
# Routes: debug introspection (only active when DEBUG_MODE=true)
# ---------------------------------------------------------------------------
def _debug_guard():
    if not DEBUG_MODE:
        return api_error("DEBUG_DISABLED", "Debug mode is disabled on this server.", 404)
    return None


@app.route('/debug/raw', methods=['GET'])
def debug_raw():
    guard = _debug_guard()
    if guard:
        return guard
    return jsonify({"raw": _last_debug["raw"], "normalized": _last_debug["normalized"]})


@app.route('/debug/chunks', methods=['GET'])
def debug_chunks():
    guard = _debug_guard()
    if guard:
        return guard
    return jsonify({"segments": _last_debug["segments"], "chunk_results": _last_debug["chunk_results"]})


@app.route('/debug/json', methods=['GET'])
def debug_json():
    guard = _debug_guard()
    if guard:
        return guard
    return jsonify({"document": _last_debug["document_json"], "timings": _last_debug["timings"]})


@app.route('/debug/html', methods=['GET'])
def debug_html():
    guard = _debug_guard()
    if guard:
        return guard
    return Response(_last_debug["html"] or "", mimetype='text/html')


# ---------------------------------------------------------------------------
# Shared request parsing for generate-pdf / structure-preview
# ---------------------------------------------------------------------------
def _parse_common_request(data: dict):
    text = data.get('text', '')
    if not isinstance(text, str) or not text.strip():
        return None, api_error("EMPTY_INPUT", "Text content is required.", 400)

    if len(text) > MAX_TEXT_CHARS:
        return None, api_error("INPUT_TOO_LARGE", f"Text too large (max {MAX_TEXT_CHARS:,} characters).", 413)

    use_ai = bool(data.get('useAI', False))
    return {"text": text, "use_ai": use_ai}, None


@app.route('/api/generate-pdf', methods=['POST'])
def generate_pdf():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return api_error("INVALID_REQUEST", "Invalid or missing JSON body.", 400)

        parsed, err = _parse_common_request(data)
        if err:
            return err
        text = parsed["text"]
        use_ai = parsed["use_ai"]

        theme_color = data.get('themeColor', '#4F46E5') or '#4F46E5'
        if not re.match(r'^#[0-9A-Fa-f]{6}$', theme_color):
            theme_color = '#4F46E5'

        try:
            font_size_pt = float(data.get('fontSizePt', 10.5))
            font_size_pt = min(max(font_size_pt, 8.0), 16.0)
        except (TypeError, ValueError):
            font_size_pt = 10.5

        page_size = data.get('pageSize', 'A4')
        if page_size not in ('A4', 'Letter'):
            page_size = 'A4'

        orientation = data.get('orientation', 'portrait')
        if orientation not in ('portrait', 'landscape'):
            orientation = 'portrait'

        title_override = data.get('title')
        if isinstance(title_override, str):
            title_override = title_override.strip()[:150] or None
        else:
            title_override = None

        images = data.get('images', [])
        images_map = {}
        if isinstance(images, list):
            if len(images) > MAX_IMAGES:
                return api_error("TOO_MANY_IMAGES", f"Too many images (max {MAX_IMAGES} per document).", 413)
            for img in images:
                if not isinstance(img, dict):
                    continue
                idx = str(img.get('index', '')).strip()
                if idx:
                    images_map[idx] = img

        theme = {
            "color": theme_color,
            "font_size_pt": font_size_pt,
            "page_size": page_size,
            "orientation": orientation,
        }

        try:
            pdf_bytes, document, timings = run_pipeline(text, use_ai, images_map, theme, title_override)
        except Exception as e:
            logger.exception("Pipeline failed")
            msg = str(e).lower()
            if 'pango' in msg or 'cairo' in msg or 'gdk' in msg or 'weasyprint' in msg:
                return api_error(
                    "RENDERER_UNAVAILABLE",
                    "PDF rendering failed due to a missing system library. Check WeasyPrint's system dependencies.",
                    500,
                )
            return api_error("PDF_GENERATION_FAILED", "PDF generation failed. Please check your input and try again.", 500)

        safe_name = re.sub(
            r'[^A-Za-z0-9_\-]+', '_', document.title or title_override or 'document'
        ).strip('_') or 'document'

        response = send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"{safe_name}.pdf"
        )
        response.headers['X-Structuring-Method'] = document.source
        response.headers['X-Total-Ms'] = str(timings.get('total_ms', ''))
        response.headers['X-Analysis-Ms'] = str(timings.get('analysis_ms', ''))
        response.headers['X-Structuring-Ms'] = str(timings.get('structuring_ms', ''))
        response.headers['X-Render-Ms'] = str(timings.get('render_ms', ''))
        response.headers['Access-Control-Expose-Headers'] = (
            'X-Structuring-Method, X-Total-Ms, X-Analysis-Ms, X-Structuring-Ms, X-Render-Ms'
        )
        return response

    except Exception:
        logger.exception("Unhandled error in generate_pdf")
        return api_error("INTERNAL_ERROR", "PDF generation failed. Please check your input and try again.", 500)


@app.route('/api/structure-preview', methods=['POST'])
def structure_preview():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return api_error("INVALID_REQUEST", "Invalid or missing JSON body.", 400)

        text = data.get('text', '')
        if not isinstance(text, str) or not text.strip():
            return jsonify({"document": None})

        if len(text) > MAX_TEXT_CHARS:
            return api_error("INPUT_TOO_LARGE", f"Text too large (max {MAX_TEXT_CHARS:,} characters).", 413)

        use_ai = bool(data.get('useAI', False))
        normalized = normalize_text(text)
        document, analysis_ms, _ = build_document(normalized, use_ai)
        return jsonify({
            "document": document_to_json(document),
            "timing": {"analysis_ms": round(analysis_ms, 1)},
        })

    except Exception:
        logger.exception("Structure preview failed")
        return api_error("INTERNAL_ERROR", "Structure preview failed.", 500)


@app.errorhandler(413)
def too_large(e):
    return api_error("PAYLOAD_TOO_LARGE", "Payload too large. Try fewer/smaller images or shorter text.", 413)


@app.errorhandler(404)
def not_found(e):
    return api_error("NOT_FOUND", "Not found.", 404)


@app.errorhandler(500)
def server_error(e):
    return api_error("INTERNAL_ERROR", "Internal server error.", 500)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=DEBUG_MODE, threaded=True)
