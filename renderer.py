"""
Renderer: Document model -> HTML+CSS -> PDF (WeasyPrint / Pango / HarfBuzz).

Why WeasyPrint instead of ReportLab's Platypus:
- Text shaping is delegated to Pango+HarfBuzz, which correctly compose
  Devanagari conjuncts and reorder matras. ReportLab's layout engine does
  not perform real text shaping, which is the root cause of broken/garbled
  Hindi & Marathi rendering in naive PDF-generation setups.
- Page numbering ("Page X of Y") and running headers are native CSS Paged
  Media features (`counter(page)`, `counter(pages)`, `string-set`), not a
  hand-rolled deferred-canvas hack.
- Structure (HTML) and presentation (CSS) are cleanly separated, matching
  the requirement that Gemini/parsing never decides colors -- only Python
  generates the CSS, using the user's theme color.

Fonts and the WeasyPrint FontConfiguration are loaded ONCE at import time
and reused for every request (this was the single biggest fixable cause of
slow, repeated per-request work in the previous implementation).
"""

import os
import io
import re
import html
import base64
import logging
import threading
from typing import Dict, List, Optional

import requests
from PIL import Image as PILImage

from document_model import Document, Block

logger = logging.getLogger("renderer")

# ---------------------------------------------------------------------------
# Fonts: one Unicode font family (Noto Sans Devanagari) covers Latin +
# Devanagari, so English/Hindi/Marathi can share a single run without
# font-switching -- the main source of broken conjuncts in mixed text.
# ---------------------------------------------------------------------------
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
# A single VARIABLE font file (wght axis 100-900) that covers both Latin
# and Devanagari. Using one variable-weight file instead of separate
# Regular/Bold statics is deliberate: Google's static/ subdirectory path
# for this family returns 404 (verified directly, not assumed), while the
# variable font path is stable. WeasyPrint/Pango support CSS Fonts 4
# variable @font-face weight ranges, so a single file still renders both
# normal and bold correctly -- see the `font-weight: 100 900;` range in
# the generated CSS below.
FONT_VARIABLE_PATH = os.path.join(FONT_DIR, "NotoSansDevanagari-Variable.ttf")
# Kept as aliases so the rest of this module can refer to "regular"/"bold"
# without caring that they resolve to the same physical file.
FONT_REGULAR_PATH = FONT_VARIABLE_PATH
FONT_BOLD_PATH = FONT_VARIABLE_PATH

FONT_URLS = {
    FONT_VARIABLE_PATH: "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansdevanagari/NotoSansDevanagari%5Bwdth%2Cwght%5D.ttf",
}

_font_lock = threading.Lock()
_fonts_ready = False
_font_config = None  # weasyprint FontConfiguration, created once


def _download_font(path: str, url: str) -> bool:
    try:
        logger.info(f"Downloading font: {url}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)
        return True
    except Exception as e:
        logger.error(f"Font download failed for {url}: {e}")
        return False


def ensure_fonts_and_engine():
    """
    Idempotent startup routine: download fonts if missing, and initialize
    a single shared WeasyPrint FontConfiguration + import weasyprint once.
    Safe to call multiple times; only does real work the first time.
    """
    global _fonts_ready, _font_config
    with _font_lock:
        if _font_config is not None:
            return

        os.makedirs(FONT_DIR, exist_ok=True)
        for path, url in FONT_URLS.items():
            if not os.path.exists(path) or os.path.getsize(path) < 1024:
                _download_font(path, url)

        _fonts_ready = os.path.exists(FONT_VARIABLE_PATH)
        if not _fonts_ready:
            logger.error(
                "Noto Sans Devanagari font missing and could not be downloaded. "
                "Hindi/Marathi text will not render correctly until fonts/ is populated. "
                "For offline environments, manually place NotoSansDevanagari-Variable.ttf "
                "in the fonts/ directory."
            )

        from weasyprint.text.fonts import FontConfiguration
        _font_config = FontConfiguration()
        logger.info("WeasyPrint font engine initialized (fonts_ready=%s).", _fonts_ready)


def fonts_ready() -> bool:
    return _fonts_ready


# ---------------------------------------------------------------------------
# Inline Markdown-ish markup (**bold**, *italic*) -> safe HTML
# ---------------------------------------------------------------------------
def _inline_markup(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', escaped)
    escaped = re.sub(r'__(.+?)__', r'<strong>\1</strong>', escaped)
    escaped = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', escaped)
    escaped = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'<em>\1</em>', escaped)
    return escaped


# ---------------------------------------------------------------------------
# Image processing: decode, fix orientation/alpha, downscale, cache per
# request so the same image referenced twice is only processed once.
# ---------------------------------------------------------------------------
def _process_image(image_obj: dict, max_px_width: int = 1400) -> Optional[str]:
    try:
        data = (image_obj or {}).get("data", "") or ""
        if data.strip().startswith("data:") and "," in data:
            data = data.split(",", 1)[1]
        raw = base64.b64decode(data)
        pil_img = PILImage.open(io.BytesIO(raw))

        if pil_img.mode == "RGBA":
            bg = PILImage.new("RGB", pil_img.size, (255, 255, 255))
            bg.paste(pil_img, mask=pil_img.split()[-1])
            pil_img = bg
        elif pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        if pil_img.width > max_px_width:
            ratio = max_px_width / float(pil_img.width)
            new_size = (max_px_width, max(1, int(pil_img.height * ratio)))
            pil_img = pil_img.resize(new_size, PILImage.LANCZOS)

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85, optimize=True)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        return None


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------
def _render_block(block: Block, images_map: Dict[str, dict], image_cache: Dict[str, Optional[str]]) -> str:
    if block.type == "heading":
        tag = f"h{block.level}"
        return f'<{tag}>{_inline_markup(block.text)}</{tag}>'

    if block.type == "paragraph":
        return f'<p>{_inline_markup(block.text)}</p>'

    if block.type == "quote":
        return f'<blockquote>{_inline_markup(block.text)}</blockquote>'

    if block.type == "bullet_list":
        items = ''.join(f'<li>{_inline_markup(i)}</li>' for i in block.items)
        return f'<ul>{items}</ul>'

    if block.type == "numbered_list":
        items = ''.join(f'<li>{_inline_markup(i)}</li>' for i in block.items)
        return f'<ol>{items}</ol>'

    if block.type == "table":
        rows_html = []
        for r_idx, row in enumerate(block.rows):
            cell_tag = "th" if r_idx == 0 else "td"
            cells = ''.join(f'<{cell_tag}>{_inline_markup(c)}</{cell_tag}>' for c in row)
            rows_html.append(f'<tr>{cells}</tr>')
        return f'<table>{"".join(rows_html)}</table>'

    if block.type == "divider":
        return '<hr>'

    if block.type == "image":
        idx = str(block.image_index)
        if idx not in image_cache:
            image_cache[idx] = _process_image(images_map.get(idx))
        data_uri = image_cache[idx]
        if data_uri:
            return f'<figure><img src="{data_uri}" alt="uploaded image"></figure>'
        return ''

    return ''


def build_html(document: Document, images_map: Dict[str, dict], theme: dict) -> str:
    """
    theme: {
        "color": "#4F46E5",
        "font_size_pt": 10.5,
        "page_size": "A4" | "Letter",
        "orientation": "portrait" | "landscape",
    }
    """
    image_cache: Dict[str, Optional[str]] = {}
    body_parts = []

    if document.title:
        body_parts.append(f'<h1 class="doc-title">{_inline_markup(document.title)}</h1>')

    for block in document.blocks:
        rendered = _render_block(block, images_map, image_cache)
        if rendered:
            body_parts.append(rendered)

    body_html = '\n'.join(body_parts) if body_parts else '<p>No content provided.</p>'
    css = _build_css(theme, document.title or "Document")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>{css}</style>
</head>
<body>
{body_html}
</body>
</html>"""


def _build_css(theme: dict, doc_title_for_footer: str) -> str:
    color = theme.get("color", "#4F46E5")
    font_size = theme.get("font_size_pt", 10.5)
    page_size = theme.get("page_size", "A4")
    orientation = theme.get("orientation", "portrait")

    size_decl = page_size
    if orientation == "landscape":
        size_decl = f"{page_size} landscape"

    font_variable_url = "file://" + FONT_VARIABLE_PATH.replace("\\", "/")

    return f"""
/* Single variable font (wght axis 100-900) covering Latin + Devanagari.
   The weight RANGE below (not a fixed value) is what tells Pango/WeasyPrint
   to interpolate within the variable font for both normal and bold text,
   rather than needing two separate font files. */
@font-face {{
    font-family: 'DocFont';
    src: url('{font_variable_url}');
    font-weight: 100 900;
}}

@page {{
    size: {size_decl};
    margin: 24mm 20mm 22mm 20mm;
    @bottom-right {{
        content: counter(page) " / " counter(pages);
        font-family: 'DocFont';
        font-size: 8pt;
        color: #888888;
    }}
    @bottom-left {{
        content: string(doctitle);
        font-family: 'DocFont';
        font-size: 8pt;
        color: #888888;
    }}
}}

* {{ box-sizing: border-box; }}

body {{
    font-family: 'DocFont', sans-serif;
    font-size: {font_size}pt;
    line-height: 1.65;
    color: #222222;
}}

h1.doc-title {{
    string-set: doctitle content();
    color: {color};
    font-size: {font_size + 11.5}pt;
    font-weight: bold;
    border-bottom: 3px solid {color};
    padding-bottom: 6px;
    margin: 0 0 14px 0;
}}

h1 {{
    string-set: doctitle content();
    color: {color};
    font-size: {font_size + 9}pt;
    font-weight: bold;
    border-bottom: 2.5px solid {color};
    padding-bottom: 5px;
    margin: 20px 0 8px 0;
    page-break-after: avoid;
}}

h2 {{
    color: {color};
    font-size: {font_size + 4.5}pt;
    font-weight: bold;
    border-bottom: 2px solid {color};
    padding-bottom: 4px;
    margin: 16px 0 6px 0;
    page-break-after: avoid;
}}

h3 {{
    color: #333333;
    font-size: {font_size + 1.5}pt;
    font-weight: bold;
    margin: 12px 0 4px 0;
    page-break-after: avoid;
}}

p {{
    margin: 0 0 8px 0;
    text-align: justify;
    orphans: 2;
    widows: 2;
}}

ul, ol {{
    margin: 4px 0 10px 0;
    padding-left: 20px;
}}

li {{
    margin-bottom: 4px;
}}

li::marker {{
    color: {color};
}}

blockquote {{
    border-left: 3px solid {color};
    margin: 10px 0;
    padding: 4px 14px;
    color: #444444;
    font-style: italic;
    background: #f7f7fb;
}}

table {{
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
    font-size: {font_size - 0.5}pt;
}}

th, td {{
    border: 1px solid #cccccc;
    padding: 6px 8px;
    text-align: left;
}}

th {{
    background: {color};
    color: #ffffff;
}}

hr {{
    border: none;
    border-top: 1.5px solid #cbd5e1;
    margin: 16px 0;
}}

figure {{
    margin: 10px 0;
    page-break-inside: avoid;
    text-align: center;
}}

figure img {{
    max-width: 100%;
    max-height: 220mm;
    height: auto;
    display: inline-block;
    border-radius: 4px;
}}
"""


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------
def render_pdf(html_string: str) -> bytes:
    ensure_fonts_and_engine()
    from weasyprint import HTML

    pdf_bytes = HTML(string=html_string).write_pdf(font_config=_font_config)
    return pdf_bytes
