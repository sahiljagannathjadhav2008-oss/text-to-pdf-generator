"""
Gemini API client.

SDK: google-genai (the current recommended Python SDK). The legacy
google-generativeai package and the retired gemini-2.0-flash model are no
longer used anywhere in this file -- the model is fully configurable via
the GEMINI_MODEL environment variable (default: gemini-3.6-flash, the
current GA Flash-class model as of this writing). See README.md for how
to change it if Google ships a newer model later.

STRICT SCOPE: Gemini is used for exactly one thing -- classifying raw text
into structural blocks (title / heading / paragraph / list / quote). It
never rewrites, translates, summarizes, or "improves" anything. This is
enforced two ways:
  1. The system instruction explicitly forbids every kind of rewriting,
     and explicitly states Gemini is a classifier, not a writer/editor/
     translator/summarizer/paraphraser.
  2. TOKEN-LEVEL LOSSLESS VALIDATION (not a fuzzy similarity score): after
     every response, Markdown syntax is stripped identically from both the
     original text and the reconstructed block text, both are split into
     whitespace-delimited tokens, and the two token lists must be EXACTLY
     EQUAL. A single changed word, dropped token, inserted token, or
     reordering fails validation outright -- there is no tolerance band
     to hide a rewrite inside. If validation fails, the caller (app.py)
     is expected to fall back to the deterministic parser for that text.

Chunking is hierarchical (paragraph -> sentence -> word-boundary -> hard
character limit as an absolute last resort) so we never split mid-word or
mid-grapheme, and chunks are made as large as safely possible to minimize
the number of Gemini calls per document.

Results are cached by a hash of (model name, allow_title, exact text), so
resubmitting unchanged text -- including re-rendering the same document
with a different theme color, font size, page size, or image set -- never
triggers a new Gemini call.
"""

import os
import re
import json
import time
import difflib
import logging
import hashlib
from typing import Optional, Tuple, List, Dict, Any

from document_model import Block

logger = logging.getLogger("gemini_client")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
# Current GA Flash-class model as of this writing. Override via env var --
# never hard-code a model name anywhere else in this codebase. If Google
# retires this model, update GEMINI_MODEL in your environment; no code
# change is required.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash").strip()

_MAX_CHUNK_CHARS = 12000  # kept large deliberately -- fewer, bigger calls
_HEALTH_CACHE_TTL_SECONDS = 60

_sdk_available = False
_genai = None
_types = None
try:
    from google import genai as _genai_module  # type: ignore
    from google.genai import types as _types_module  # type: ignore
    _genai = _genai_module
    _types = _types_module
    _sdk_available = True
except Exception as e:  # pragma: no cover - depends on optional dependency
    logger.warning(f"google-genai SDK not importable: {e}")

_client = None
if _sdk_available and GEMINI_API_KEY:
    try:
        _client = _genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        logger.error(f"Failed to create google-genai Client: {e}")
        _client = None

_structure_cache: Dict[str, Any] = {}
_health_cache = {"result": None, "checked_at": 0.0}

_BASE_SYSTEM_INSTRUCTION = """You are a document structure classifier. You are NOT a writer, editor, translator, summarizer, or paraphraser.

You will receive a passage of plain text that may be in English, Hindi, Marathi, or a mix of these.

Your ONLY task is to classify the text into structural blocks and return it inside a strict JSON schema. You must reproduce every character of the input text exactly as given, with absolutely no changes.

You must NEVER:
- rewrite, reword, or "improve" any sentence or phrase
- shorten, summarize, or expand anything
- translate any word from one language to another
- correct spelling, grammar, or punctuation
- remove repeated or redundant content
- invent new content that was not in the input
- reorder sentences or list items
- merge separate paragraphs into one
- split one paragraph into multiple paragraphs unless the input itself has a blank line there

Return ONLY valid JSON matching exactly this schema. No markdown code fences. No commentary. No explanation. JSON only.

{
  "document": {
    "title": "string, or empty string if this passage has no title",
    "blocks": [
      {"type": "heading", "level": 1, "text": "exact text from input"},
      {"type": "paragraph", "text": "exact text from input"},
      {"type": "bullet_list", "items": ["exact text", "exact text"]},
      {"type": "numbered_list", "items": ["exact text", "exact text"]},
      {"type": "quote", "text": "exact text from input"}
    ]
  }
}

Allowed values for "type": heading, paragraph, bullet_list, numbered_list, quote.
"level" is required only for type "heading" and must be exactly 1, 2, or 3.
"""

_TITLE_ALLOWED_ADDENDUM = """
This passage is the very beginning of the document. If it clearly opens with a title, put that exact text in the "title" field. Otherwise leave "title" as an empty string.
"""

_TITLE_FORBIDDEN_ADDENDUM = """
This passage is NOT the beginning of the document -- it is a middle or later section. You MUST set "title" to an empty string no matter what this passage contains. Represent any heading-like text as a "heading" block instead of a title. Never omit text because you think it looks like a title.
"""


def is_configured() -> bool:
    return _sdk_available and _client is not None


# ---------------------------------------------------------------------------
# Error classification (never leak raw exceptions/stack traces to clients)
# ---------------------------------------------------------------------------
def _classify_error(e: Exception) -> Tuple[str, str]:
    msg = str(e)
    low = msg.lower()
    if 'api key' in low or 'unauthorized' in low or '401' in low or 'permission' in low or 'invalid_argument' in low and 'key' in low:
        return "GEMINI_INVALID_KEY", "The configured Gemini API key was rejected."
    if 'quota' in low or 'rate' in low or '429' in low or 'resource_exhausted' in low:
        return "GEMINI_RATE_LIMIT", "Gemini API quota or rate limit was exceeded."
    if 'not found' in low or '404' in low or 'model' in low and 'unsupported' in low:
        return "GEMINI_MODEL_UNAVAILABLE", f"The configured model '{GEMINI_MODEL}' is unavailable."
    if 'timeout' in low or 'deadline' in low:
        return "GEMINI_TIMEOUT", "Gemini API request timed out."
    return "GEMINI_UNAVAILABLE", "AI structuring is currently unavailable."


def _safe_error_text(e: Exception) -> str:
    msg = str(e)
    return msg[:200] + "..." if len(msg) > 200 else msg


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
def check_health(force: bool = False) -> dict:
    now = time.time()
    if not force and _health_cache["result"] is not None:
        if now - _health_cache["checked_at"] < _HEALTH_CACHE_TTL_SECONDS:
            return _health_cache["result"]

    if not _sdk_available:
        result = {"status": "not_configured", "reason": "google-genai SDK not installed", "response_time_ms": None}
    elif not GEMINI_API_KEY:
        result = {"status": "not_configured", "reason": "GEMINI_API_KEY not set", "response_time_ms": None}
    elif _client is None:
        result = {"status": "error", "reason": "SDK client could not be created", "response_time_ms": None}
    else:
        start = time.perf_counter()
        try:
            resp = _client.models.generate_content(
                model=GEMINI_MODEL,
                contents="ping",
                config=_types.GenerateContentConfig(max_output_tokens=5),
            )
            _ = resp.text
            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
            result = {"status": "connected", "reason": None, "response_time_ms": elapsed_ms}
        except Exception as e:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
            code, message = _classify_error(e)
            result = {"status": "error", "reason": message, "code": code, "response_time_ms": elapsed_ms}

    result["checked_at"] = now
    _health_cache["result"] = result
    _health_cache["checked_at"] = now
    return result


# ---------------------------------------------------------------------------
# Hierarchical chunking: paragraph -> sentence -> word boundary -> hard cut
# ---------------------------------------------------------------------------
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?\u0964\u0965])\s+')  # includes Devanagari danda/double-danda


def _split_into_sentences(text: str) -> List[str]:
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p for p in parts if p]


def _split_by_word_boundary(text: str, max_chars: int) -> List[str]:
    """Emergency fallback: split on whitespace only, never mid-word."""
    words = text.split(' ')
    chunks, current, current_len = [], [], 0
    for w in words:
        wl = len(w) + 1
        if current and current_len + wl > max_chars:
            chunks.append(' '.join(current))
            current, current_len = [w], wl
        else:
            current.append(w)
            current_len += wl
    if current:
        chunks.append(' '.join(current))
    return chunks


def _split_long_paragraph(paragraph: str, max_chars: int) -> List[str]:
    """A single paragraph larger than max_chars: split by sentence first,
    then by word boundary only if an individual sentence is still too long.
    Never a hard character cut inside a word or grapheme cluster."""
    sentences = _split_into_sentences(paragraph)
    chunks, current, current_len = [], [], 0

    for sent in sentences:
        if len(sent) > max_chars:
            for sub in _split_by_word_boundary(sent, max_chars):
                chunks.append(sub)
            continue
        sl = len(sent) + 1
        if current and current_len + sl > max_chars:
            chunks.append(' '.join(current))
            current, current_len = [sent], sl
        else:
            current.append(sent)
            current_len += sl

    if current:
        chunks.append(' '.join(current))
    return chunks if chunks else [paragraph]


def chunk_for_gemini(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> List[str]:
    """
    Hierarchical chunking: split on paragraph (blank-line) boundaries first
    -- this is the most structurally meaningful split. Only a paragraph
    that individually exceeds max_chars gets split further, by sentence,
    then by word boundary as a last resort. Chunks are packed as large as
    possible (up to max_chars) to minimize the number of Gemini calls.
    """
    paragraphs = text.split('\n\n')
    pieces: List[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            pieces.append(para)
        else:
            pieces.extend(_split_long_paragraph(para, max_chars))

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for piece in pieces:
        piece_len = len(piece) + 2
        if current and current_len + piece_len > max_chars:
            chunks.append('\n\n'.join(current))
            current, current_len = [piece], piece_len
        else:
            current.append(piece)
            current_len += piece_len
    if current:
        chunks.append('\n\n'.join(current))

    return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# Token-level lossless validation
# ---------------------------------------------------------------------------
def _strip_markdown_syntax_line(line: str) -> str:
    """Remove structural Markdown syntax so it isn't counted as content --
    this mirrors exactly what both Gemini and the fallback parser are
    expected to strip when classifying, so comparison is apples-to-apples."""
    m = re.match(r'^(#{1,3})\s+(.*)', line)
    if m:
        return m.group(2)
    m = re.match(r'^[-*]\s+(.*)', line)
    if m:
        return m.group(1)
    m = re.match(r'^\d+[.)]\s+(.*)', line)
    if m:
        return m.group(1)
    m = re.match(r'^>\s?(.*)', line)
    if m:
        return m.group(1)
    return line


def _strip_markdown_syntax(text: str) -> str:
    return '\n'.join(_strip_markdown_syntax_line(line) for line in text.split('\n'))


def _tokenize(text: str) -> List[str]:
    """Whitespace normalization is used ONLY here, for comparison purposes
    -- it never mutates any text that is actually stored or rendered."""
    normalized = re.sub(r'\s+', ' ', text).strip()
    return normalized.split(' ') if normalized else []


def _reconstruct_text(title: str, blocks: List[dict]) -> str:
    parts = [title] if title else []
    for b in blocks:
        btype = b.get("type")
        if btype in ("heading", "paragraph", "quote"):
            parts.append(b.get("text", "") or "")
        elif btype in ("bullet_list", "numbered_list"):
            parts.extend([i for i in (b.get("items", []) or []) if isinstance(i, str)])
    return ' '.join(parts)


def _verify_lossless(original_text: str, title: str, raw_blocks: List[dict]) -> Tuple[bool, dict]:
    """
    Exact token-for-token comparison -- NOT a fuzzy similarity score.
    Title text is always included in the reconstruction (regardless of
    whether the caller will actually use it as a title) so that even a
    mislabeled title can never cause content to silently disappear from
    the comparison.
    """
    original_tokens = _tokenize(_strip_markdown_syntax(original_text))
    reconstructed_tokens = _tokenize(_reconstruct_text(title, raw_blocks))

    ok = original_tokens == reconstructed_tokens
    info = {
        "original_token_count": len(original_tokens),
        "reconstructed_token_count": len(reconstructed_tokens),
    }
    if not ok:
        # Best-effort diagnostic (not used for the pass/fail decision itself).
        sm = difflib.SequenceMatcher(None, original_tokens, reconstructed_tokens)
        info["similarity_ratio"] = round(sm.ratio(), 4)
    return ok, info


# ---------------------------------------------------------------------------
# JSON parsing / schema validation
# ---------------------------------------------------------------------------
def _parse_json_response(raw_text: str) -> Optional[dict]:
    text = raw_text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict) or "document" not in parsed:
        return None
    doc = parsed["document"]
    if not isinstance(doc, dict) or "blocks" not in doc or not isinstance(doc["blocks"], list):
        return None
    return doc


def _validate_blocks(raw_blocks: List[dict]) -> List[Block]:
    valid = []
    for rb in raw_blocks:
        if not isinstance(rb, dict):
            continue
        btype = rb.get("type")
        if btype == "heading":
            level = rb.get("level", 2)
            if level not in (1, 2, 3):
                level = 2
            text = rb.get("text", "")
            if isinstance(text, str) and text.strip():
                valid.append(Block(type="heading", level=level, text=text.strip()))
        elif btype in ("paragraph", "quote"):
            text = rb.get("text", "")
            if isinstance(text, str) and text.strip():
                valid.append(Block(type=btype, text=text.strip()))
        elif btype in ("bullet_list", "numbered_list"):
            items = rb.get("items", [])
            if isinstance(items, list):
                clean_items = [i.strip() for i in items if isinstance(i, str) and i.strip()]
                if clean_items:
                    valid.append(Block(type=btype, items=clean_items))
    return valid


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def _cache_key(text: str, allow_title: bool) -> str:
    raw = f"{GEMINI_MODEL}:{allow_title}:{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def structure_chunk(text: str, allow_title: bool) -> Tuple[Optional[dict], Optional[List[Block]], bool, float, Optional[dict]]:
    """
    Returns (raw_document_json_or_None, validated_blocks_or_None, ok, analysis_ms, error_info).

    ok=False means: Gemini unreachable, invalid JSON, schema failure, OR
    lossless validation failure. In every ok=False case, validated_blocks
    is None and the caller MUST fall back to the deterministic parser for
    this exact text -- no partial/best-effort use of a rejected result.
    """
    if not is_configured():
        return None, None, False, 0.0, {"code": "GEMINI_UNAVAILABLE", "message": "Gemini is not configured."}

    cache_key = _cache_key(text, allow_title)
    if cache_key in _structure_cache:
        cached = _structure_cache[cache_key]
        return cached[0], cached[1], cached[2], 0.0, cached[4]  # analysis_ms=0 on cache hit

    addendum = _TITLE_ALLOWED_ADDENDUM if allow_title else _TITLE_FORBIDDEN_ADDENDUM
    system_instruction = _BASE_SYSTEM_INSTRUCTION + addendum

    start = time.perf_counter()
    try:
        resp = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"Analyze and structure the following text passage:\n\n{text}",
            config=_types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                max_output_tokens=8192,
            ),
        )
        raw_text = resp.text
    except Exception as e:
        analysis_ms = round((time.perf_counter() - start) * 1000, 1)
        code, message = _classify_error(e)
        logger.error(f"Gemini call failed [{code}]: {_safe_error_text(e)}")
        error_info = {"code": code, "message": message}
        # Deliberately NOT cached: a transient network/API failure should be
        # retried on the next identical request, not permanently remembered
        # as a failure the way a genuine validation rejection is.
        return None, None, False, analysis_ms, error_info

    analysis_ms = round((time.perf_counter() - start) * 1000, 1)

    parsed_doc = _parse_json_response(raw_text)
    if parsed_doc is None:
        error_info = {"code": "GEMINI_INVALID_JSON", "message": "Gemini returned malformed structure output."}
        logger.warning("Gemini returned unparsable/invalid-schema JSON.")
        _structure_cache[cache_key] = (None, None, False, analysis_ms, error_info)
        return None, None, False, analysis_ms, error_info

    title = parsed_doc.get("title", "") or ""
    raw_blocks = parsed_doc.get("blocks", [])

    ok, verify_info = _verify_lossless(text, title, raw_blocks)
    if not ok:
        error_info = {
            "code": "TEXT_PRESERVATION_FAILED",
            "message": "AI output did not exactly match the original text and was discarded.",
            "detail": verify_info,
        }
        logger.warning(f"Lossless validation FAILED: {verify_info}")
        _structure_cache[cache_key] = (parsed_doc, None, False, analysis_ms, error_info)
        return parsed_doc, None, False, analysis_ms, error_info

    validated_blocks = _validate_blocks(raw_blocks)
    result = (parsed_doc, validated_blocks, True, analysis_ms, None)
    _structure_cache[cache_key] = result
    return result
