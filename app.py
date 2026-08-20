"""
AI Text to Organized Multi-Language PDF Generator
====================================================
Backend: Flask + ReportLab (Platypus framework)

Features:
- Full Unicode support for English, Hindi, and Marathi (Devanagari script)
  via the Noto Sans Devanagari font family (auto-downloaded on first run).
- Smart Markdown-style parsing: #, ##, ### headings, -/* bullet lists,
  1. numbered lists, --- horizontal rules, **bold**/*italic* inline markup.
- Streams 100+ page documents efficiently using ReportLab's Platypus
  flowable model (constant memory footprint regardless of document size).
- Dynamic "Page X of Y" footer via a custom Canvas subclass.
- Theme color (hex) applied to headings and decorative accent lines.
- Image insertion via [[image:N]] markers, auto-scaled to fit content width.

Run:
    pip install -r requirements.txt
    python app.py

Deploy (example, Render/Railway/Fly.io/etc.):
    gunicorn app:app --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT
"""

import os
import re
import io
import base64
import logging
import threading

import requests
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    ListFlowable, ListItem, KeepTogether, Flowable
)
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pdfgen")

app = Flask(__name__)
CORS(app)  # Allow GitHub Pages (or any) frontend origin. Restrict in prod if desired.
app.config['MAX_CONTENT_LENGTH'] = 60 * 1024 * 1024  # 60 MB request cap (covers many embedded images)

# ---------------------------------------------------------------------------
# Font handling: Noto Sans Devanagari covers Latin + Devanagari glyphs,
# so a SINGLE font family renders English, Hindi, and Marathi correctly
# in the same paragraph without font-switching hacks or missing-glyph boxes.
# ---------------------------------------------------------------------------
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
FONT_REGULAR_PATH = os.path.join(FONT_DIR, "NotoSansDevanagari-Regular.ttf")
FONT_BOLD_PATH = os.path.join(FONT_DIR, "NotoSansDevanagari-Bold.ttf")

FONT_URLS = {
    FONT_REGULAR_PATH: "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansdevanagari/static/NotoSansDevanagari-Regular.ttf",
    FONT_BOLD_PATH: "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansdevanagari/static/NotoSansDevanagari-Bold.ttf",
}

FONT_NAME = "NotoSansDevanagari"
FONT_NAME_BOLD = "NotoSansDevanagari-Bold"

_font_lock = threading.Lock()
_fonts_ready = False


def _download_font(path, url):
    """Download a font file if not already present locally."""
    try:
        logger.info(f"Downloading font from {url} ...")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)
        logger.info(f"Saved font to {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download font {url}: {e}")
        return False


def ensure_fonts():
    """
    Idempotent, thread-safe font bootstrap. Downloads Noto Sans Devanagari
    (Regular + Bold) into ./fonts if missing, then registers them with
    ReportLab. Falls back to core Helvetica (Latin-only) if unavailable,
    so the server never crashes even without internet access -- though
    Hindi/Marathi text will not render correctly in that fallback case.

    For fully offline deployments, manually place the two TTF files in
    the ./fonts directory before starting the server (same filenames as
    above) and this function will detect and use them without any
    network call.
    """
    global _fonts_ready
    with _font_lock:
        if _fonts_ready:
            return
        os.makedirs(FONT_DIR, exist_ok=True)

        for path, url in FONT_URLS.items():
            if not os.path.exists(path) or os.path.getsize(path) < 1024:
                _download_font(path, url)

        try:
            if not os.path.exists(FONT_REGULAR_PATH):
                raise FileNotFoundError(FONT_REGULAR_PATH)

            pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_REGULAR_PATH))

            if os.path.exists(FONT_BOLD_PATH):
                pdfmetrics.registerFont(TTFont(FONT_NAME_BOLD, FONT_BOLD_PATH))
            else:
                # Bold fallback: reuse regular glyphs if bold file missing
                pdfmetrics.registerFont(TTFont(FONT_NAME_BOLD, FONT_REGULAR_PATH))

            registerFontFamily(
                FONT_NAME,
                normal=FONT_NAME,
                bold=FONT_NAME_BOLD,
                italic=FONT_NAME,
                boldItalic=FONT_NAME_BOLD,
            )
            _fonts_ready = True
            logger.info("Unicode fonts registered successfully (Noto Sans Devanagari).")
        except Exception as e:
            _fonts_ready = False
            logger.error(
                f"Could not register Unicode fonts, falling back to Helvetica "
                f"(Hindi/Marathi glyphs will NOT render correctly): {e}"
            )


# ---------------------------------------------------------------------------
# Decorative accent line flowable (used under headings and as an HR)
# ---------------------------------------------------------------------------
class AccentLine(Flowable):
    def __init__(self, width=140, thickness=2.6, color="#4F46E5"):
        super().__init__()
        self.width = width
        self.thickness = thickness
        self.color = HexColor(color)
        self.height = thickness + 8

    def draw(self):
        self.canv.saveState()
        self.canv.setFillColor(self.color)
        self.canv.roundRect(0, 5, self.width, self.thickness, radius=self.thickness / 2, stroke=0, fill=1)
        self.canv.restoreState()

    def wrap(self, availWidth, availHeight):
        return (self.width, self.height)


# ---------------------------------------------------------------------------
# Custom canvas: page numbering ("Page X of Y") + running header/footer
# ---------------------------------------------------------------------------
def make_canvas_class(theme_hex, doc_title):
    theme_color = HexColor(theme_hex)

    class NumberedCanvas(pdfcanvas.Canvas):
        def __init__(self, *args, **kwargs):
            pdfcanvas.Canvas.__init__(self, *args, **kwargs)
            self._saved_page_states = []

        def showPage(self):
            # Defer actual page rendering until we know the final page count
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            total_pages = len(self._saved_page_states)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                self._draw_page_furniture(total_pages)
                pdfcanvas.Canvas.showPage(self)
            pdfcanvas.Canvas.save(self)

        def _draw_page_furniture(self, total_pages):
            page_w, page_h = A4
            font = FONT_NAME if _fonts_ready else 'Helvetica'

            # Top accent bar
            self.saveState()
            self.setFillColor(theme_color)
            self.rect(0, page_h - 5 * mm, page_w, 5 * mm, stroke=0, fill=1)
            self.restoreState()

            # Footer divider line
            self.saveState()
            self.setStrokeColor(HexColor('#DDDDDD'))
            self.setLineWidth(0.6)
            self.line(20 * mm, 17 * mm, page_w - 20 * mm, 17 * mm)

            # Footer left: document title
            self.setFont(font, 8)
            self.setFillColor(HexColor('#888888'))
            title_text = (doc_title or "")[:70]
            self.drawString(20 * mm, 11 * mm, title_text)

            # Footer right: Page X of Y
            self.setFillColor(theme_color)
            self.setFont(font, 8)
            page_label = f"Page {self._pageNumber} of {total_pages}"
            self.drawRightString(page_w - 20 * mm, 11 * mm, page_label)
            self.restoreState()

    return NumberedCanvas


# ---------------------------------------------------------------------------
# Paragraph styles (theme-aware)
# ---------------------------------------------------------------------------
def build_styles(theme_hex):
    base_font = FONT_NAME if _fonts_ready else 'Helvetica'
    bold_font = FONT_NAME_BOLD if _fonts_ready else 'Helvetica-Bold'
    theme = HexColor(theme_hex)

    return {
        'H1': ParagraphStyle(
            'H1Style', fontName=bold_font, fontSize=22, leading=28,
            textColor=theme, spaceBefore=20, spaceAfter=2, alignment=TA_LEFT
        ),
        'H2': ParagraphStyle(
            'H2Style', fontName=bold_font, fontSize=16, leading=21,
            textColor=theme, spaceBefore=16, spaceAfter=2, alignment=TA_LEFT
        ),
        'H3': ParagraphStyle(
            'H3Style', fontName=bold_font, fontSize=13, leading=17,
            textColor=HexColor('#333333'), spaceBefore=12, spaceAfter=3, alignment=TA_LEFT
        ),
        'Body': ParagraphStyle(
            'BodyStyle', fontName=base_font, fontSize=10.5, leading=16.5,
            textColor=HexColor('#222222'), alignment=TA_JUSTIFY, spaceAfter=4
        ),
        'Bullet': ParagraphStyle(
            'BulletStyle', fontName=base_font, fontSize=10.5, leading=15.5,
            textColor=HexColor('#222222'), alignment=TA_LEFT
        ),
    }


# ---------------------------------------------------------------------------
# Inline Markdown -> ReportLab mini-markup (bold / italic), XML-safe
# ---------------------------------------------------------------------------
def escape_xml(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def inline_markup(s):
    s = escape_xml(s)
    s = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s)
    s = re.sub(r'__(.+?)__', r'<b>\1</b>', s)
    s = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', s)
    s = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'<i>\1</i>', s)
    return s


# ---------------------------------------------------------------------------
# Image flowable builder: decodes base64, preserves aspect ratio, caps size
# ---------------------------------------------------------------------------
def build_image_flowable(image_obj, max_width, max_height=420):
    if not image_obj:
        return None
    try:
        data = image_obj.get('data', '') or ''
        if data.strip().startswith('data:') and ',' in data:
            data = data.split(',', 1)[1]

        raw = base64.b64decode(data)
        pil_img = PILImage.open(io.BytesIO(raw))

        if pil_img.mode == 'RGBA':
            bg = PILImage.new('RGB', pil_img.size, (255, 255, 255))
            bg.paste(pil_img, mask=pil_img.split()[-1])
            pil_img = bg
        elif pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')

        iw, ih = pil_img.size
        if iw <= 0 or ih <= 0:
            return None

        scale = min(max_width / float(iw), max_height / float(ih))
        new_w = max(1, iw * scale)
        new_h = max(1, ih * scale)

        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=87, optimize=True)
        buf.seek(0)

        img_flowable = RLImage(buf, width=new_w, height=new_h)
        return KeepTogether([Spacer(1, 8), img_flowable, Spacer(1, 10)])
    except Exception as e:
        logger.error(f"Failed to process embedded image: {e}")
        return None


# ---------------------------------------------------------------------------
# Markdown -> Platypus story builder
# ---------------------------------------------------------------------------
def build_story(text, images_map, styles, theme_hex, content_width):
    story = []
    lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')

    bullet_buffer = []
    numbered_buffer = []

    def flush_bullets():
        nonlocal bullet_buffer
        if bullet_buffer:
            items = [ListItem(Paragraph(t, styles['Bullet']), leftIndent=6, spaceAfter=3) for t in bullet_buffer]
            story.append(ListFlowable(
                items, bulletType='bullet', start='circle',
                leftIndent=18, bulletFontSize=7, bulletColor=HexColor(theme_hex)
            ))
            story.append(Spacer(1, 8))
            bullet_buffer = []

    def flush_numbered():
        nonlocal numbered_buffer
        if numbered_buffer:
            items = [ListItem(Paragraph(t, styles['Bullet']), leftIndent=6, spaceAfter=3) for t in numbered_buffer]
            story.append(ListFlowable(
                items, bulletType='1', start='1',
                leftIndent=18, bulletFontSize=9.5, bulletColor=HexColor(theme_hex)
            ))
            story.append(Spacer(1, 8))
            numbered_buffer = []

    def flush_all():
        flush_bullets()
        flush_numbered()

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_all()
            story.append(Spacer(1, 6))
            continue

        img_match = re.match(r'^\[\[image:(\d+)\]\]$', stripped)
        if img_match:
            flush_all()
            flow = build_image_flowable(images_map.get(img_match.group(1)), content_width)
            if flow:
                story.append(flow)
            continue

        if stripped in ('---', '***', '___'):
            flush_all()
            story.append(Spacer(1, 6))
            story.append(AccentLine(width=content_width, thickness=1, color='#CBD5E1'))
            story.append(Spacer(1, 10))
            continue

        if stripped.startswith('### '):
            flush_all()
            story.append(Paragraph(inline_markup(stripped[4:].strip()), styles['H3']))
            continue

        if stripped.startswith('## '):
            flush_all()
            story.append(Paragraph(inline_markup(stripped[3:].strip()), styles['H2']))
            story.append(AccentLine(width=70, thickness=2.4, color=theme_hex))
            story.append(Spacer(1, 6))
            continue

        if stripped.startswith('# '):
            flush_all()
            story.append(Paragraph(inline_markup(stripped[2:].strip()), styles['H1']))
            story.append(AccentLine(width=110, thickness=3, color=theme_hex))
            story.append(Spacer(1, 8))
            continue

        bullet_match = re.match(r'^[-*]\s+(.*)', stripped)
        if bullet_match:
            flush_numbered()
            bullet_buffer.append(inline_markup(bullet_match.group(1)))
            continue

        numbered_match = re.match(r'^\d+[.)]\s+(.*)', stripped)
        if numbered_match:
            flush_bullets()
            numbered_buffer.append(inline_markup(numbered_match.group(1)))
            continue

        flush_all()
        story.append(Paragraph(inline_markup(stripped), styles['Body']))

    flush_all()
    return story


# ---------------------------------------------------------------------------
# PDF assembly
# ---------------------------------------------------------------------------
def build_pdf(text, images_map, theme_hex, title):
    ensure_fonts()

    buf = io.BytesIO()
    page_w, page_h = A4
    margin = 20 * mm

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin + 8,
        bottomMargin=margin + 6,
        title=title or "Document",
        author="AI PDF Generator",
    )

    content_width = page_w - 2 * margin
    styles = build_styles(theme_hex)
    story = build_story(text, images_map, styles, theme_hex, content_width)

    if not story:
        story = [Paragraph("No content provided.", styles['Body'])]

    canvas_cls = make_canvas_class(theme_hex, title or "Document")
    doc.build(story, canvasmaker=canvas_cls)

    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "fonts_ready": _fonts_ready})


@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "Invalid or missing JSON body"}), 400

        text = data.get('text', '')
        if not text or not text.strip():
            return jsonify({"error": "Text content is required"}), 400

        if len(text) > 2_000_000:
            return jsonify({"error": "Text too large (max ~2,000,000 characters)"}), 413

        theme_hex = data.get('themeColor', '#4F46E5') or '#4F46E5'
        if not re.match(r'^#[0-9A-Fa-f]{6}$', theme_hex):
            theme_hex = '#4F46E5'

        title = (data.get('title') or 'Document').strip()[:120]

        images = data.get('images', [])
        images_map = {}
        if isinstance(images, list):
            for img in images:
                if not isinstance(img, dict):
                    continue
                idx = str(img.get('index', '')).strip()
                if idx:
                    images_map[idx] = img

        pdf_buffer = build_pdf(text, images_map, theme_hex, title)

        safe_name = re.sub(r'[^A-Za-z0-9_\-]+', '_', title).strip('_') or 'document'
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"{safe_name}.pdf"
        )

    except Exception as e:
        logger.exception("PDF generation failed")
        return jsonify({"error": f"PDF generation failed: {str(e)}"}), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "Payload too large. Try fewer/smaller images."}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    ensure_fonts()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
