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
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    ListFlowable,
    ListItem,
    KeepTogether,
    HRFlowable,
    Table,
    TableStyle,
)
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from PIL import Image as PILImage


# ============================================================
# APP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("text-to-pdf")

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024


# ============================================================
# FONTS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(BASE_DIR, "fonts")

REGULAR_FONT = os.path.join(
    FONT_DIR,
    "NotoSansDevanagari-Regular.ttf"
)

BOLD_FONT = os.path.join(
    FONT_DIR,
    "NotoSansDevanagari-Bold.ttf"
)

FONT_REGULAR_NAME = "NotoDevanagari"
FONT_BOLD_NAME = "NotoDevanagari-Bold"

# Static Devanagari UI fonts
FONT_URLS = {
    REGULAR_FONT:
        "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansdevanagariui/NotoSansDevanagariUI-Regular.ttf",

    BOLD_FONT:
        "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansdevanagariui/NotoSansDevanagariUI-Bold.ttf"
}

_fonts_ready = False
_font_lock = threading.Lock()


def download_font(url, path):

    try:

        logger.info("Downloading font: %s", url)

        response = requests.get(
            url,
            timeout=60,
            headers={
                "User-Agent": "SJ-Text-to-PDF"
            }
        )

        response.raise_for_status()

        if len(response.content) < 10000:
            raise ValueError("Invalid font file")

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True
        )

        with open(path, "wb") as file:
            file.write(response.content)

        return True

    except Exception as error:

        logger.error(
            "Font download failed: %s",
            error
        )

        return False


def ensure_fonts():

    global _fonts_ready

    with _font_lock:

        if _fonts_ready:
            return True

        os.makedirs(
            FONT_DIR,
            exist_ok=True
        )

        # Download only if missing
        for path, url in FONT_URLS.items():

            if (
                not os.path.exists(path)
                or os.path.getsize(path) < 10000
            ):

                download_font(
                    url,
                    path
                )

        try:

            if not os.path.exists(REGULAR_FONT):
                raise FileNotFoundError(
                    REGULAR_FONT
                )

            pdfmetrics.registerFont(
                TTFont(
                    FONT_REGULAR_NAME,
                    REGULAR_FONT
                )
            )

            if os.path.exists(BOLD_FONT):

                pdfmetrics.registerFont(
                    TTFont(
                        FONT_BOLD_NAME,
                        BOLD_FONT
                    )
                )

            else:

                pdfmetrics.registerFont(
                    TTFont(
                        FONT_BOLD_NAME,
                        REGULAR_FONT
                    )
                )

            registerFontFamily(
                FONT_REGULAR_NAME,
                normal=FONT_REGULAR_NAME,
                bold=FONT_BOLD_NAME,
                italic=FONT_REGULAR_NAME,
                boldItalic=FONT_BOLD_NAME
            )

            _fonts_ready = True

            logger.info(
                "Unicode font loaded successfully."
            )

            return True

        except Exception as error:

            _fonts_ready = False

            logger.error(
                "Font registration failed: %s",
                error
            )

            return False


def normal_font():

    return (
        FONT_REGULAR_NAME
        if _fonts_ready
        else "Helvetica"
    )


def bold_font():

    return (
        FONT_BOLD_NAME
        if _fonts_ready
        else "Helvetica-Bold"
    )


# ============================================================
# THEMES
# ============================================================

THEMES = {

    "blue": "#2563EB",

    "indigo": "#4F46E5",

    "purple": "#7C3AED",

    "green": "#059669",

    "teal": "#0F766E",

    "orange": "#EA580C",

    "red": "#DC2626",

    "pink": "#DB2777",

    "slate": "#334155",

    "black": "#111827"
}


def clean_theme(value):

    if not isinstance(value, str):
        return "#2563EB"

    value = value.strip()

    # #RRGGBB
    if re.fullmatch(
        r"#[0-9A-Fa-f]{6}",
        value
    ):
        return value.upper()

    # RRGGBB
    if re.fullmatch(
        r"[0-9A-Fa-f]{6}",
        value
    ):
        return "#" + value.upper()

    # Theme name
    return THEMES.get(
        value.lower(),
        "#2563EB"
    )


def light_color(hex_color):

    value = hex_color.replace("#", "")

    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)

    r = int(r + (255 - r) * 0.90)
    g = int(g + (255 - g) * 0.90)
    b = int(b + (255 - b) * 0.90)

    return (
        f"#{r:02X}"
        f"{g:02X}"
        f"{b:02X}"
    )


# ============================================================
# PAGE CANVAS
# ============================================================

def make_canvas(theme_hex, title):

    theme = HexColor(theme_hex)

    class DocumentCanvas(pdfcanvas.Canvas):

        def __init__(
            self,
            *args,
            **kwargs
        ):

            super().__init__(
                *args,
                **kwargs
            )

            self.saved_states = []

        def showPage(self):

            self.saved_states.append(
                dict(self.__dict__)
            )

            self._startPage()

        def save(self):

            total_pages = len(
                self.saved_states
            )

            for state in self.saved_states:

                self.__dict__.update(
                    state
                )

                self.draw_page(
                    total_pages
                )

                pdfcanvas.Canvas.showPage(
                    self
                )

            pdfcanvas.Canvas.save(
                self
            )

        def draw_page(self, total):

            page_width, page_height = A4

            self.saveState()

            # Top theme strip
            self.setFillColor(theme)

            self.rect(
                0,
                page_height - 4 * mm,
                page_width,
                4 * mm,
                stroke=0,
                fill=1
            )

            # Footer line
            self.setStrokeColor(
                HexColor("#E5E7EB")
            )

            self.setLineWidth(0.6)

            self.line(
                20 * mm,
                17 * mm,
                page_width - 20 * mm,
                17 * mm
            )

            # Footer title
            self.setFont(
                normal_font(),
                7.8
            )

            self.setFillColor(
                HexColor("#64748B")
            )

            footer_title = (
                title or "Document"
            ).replace(
                "\n",
                " "
            )[:65]

            self.drawString(
                20 * mm,
                11 * mm,
                footer_title
            )

            # Page number
            self.setFillColor(theme)

            self.drawRightString(
                page_width - 20 * mm,
                11 * mm,
                f"Page {self._pageNumber} of {total}"
            )

            self.restoreState()

    return DocumentCanvas


# ============================================================
# STYLES
# ============================================================

def build_styles(theme_hex):

    theme = HexColor(theme_hex)

    pale = HexColor(
        light_color(theme_hex)
    )

    return {

        "TITLE": ParagraphStyle(
            "TITLE",

            fontName=bold_font(),

            fontSize=25,

            leading=31,

            textColor=theme,

            spaceBefore=2,

            spaceAfter=8,

            keepWithNext=True
        ),

        "H1": ParagraphStyle(
            "H1",

            fontName=bold_font(),

            fontSize=20,

            leading=26,

            textColor=theme,

            spaceBefore=18,

            spaceAfter=5,

            keepWithNext=True
        ),

        "H2": ParagraphStyle(
            "H2",

            fontName=bold_font(),

            fontSize=15.5,

            leading=21,

            textColor=theme,

            spaceBefore=15,

            spaceAfter=4,

            keepWithNext=True
        ),

        "H3": ParagraphStyle(
            "H3",

            fontName=bold_font(),

            fontSize=12.5,

            leading=17,

            textColor=HexColor("#1F2937"),

            spaceBefore=10,

            spaceAfter=4,

            keepWithNext=True
        ),

        "BODY": ParagraphStyle(
            "BODY",

            fontName=normal_font(),

            fontSize=10.5,

            leading=16.2,

            textColor=HexColor("#1F2937"),

            alignment=TA_JUSTIFY,

            spaceAfter=7
        ),

        "BULLET": ParagraphStyle(
            "BULLET",

            fontName=normal_font(),

            fontSize=10.3,

            leading=15.5,

            textColor=HexColor("#1F2937"),

            spaceAfter=2
        ),

        "QUOTE": ParagraphStyle(
            "QUOTE",

            fontName=normal_font(),

            fontSize=10.3,

            leading=15.5,

            textColor=HexColor("#475569"),

            leftIndent=10,

            rightIndent=6,

            borderColor=theme,

            borderWidth=1,

            borderPadding=7,

            backColor=pale,

            spaceBefore=5,

            spaceAfter=8
        ),

        "TABLE": ParagraphStyle(
            "TABLE",

            fontName=normal_font(),

            fontSize=9.2,

            leading=12.5,

            textColor=HexColor("#1F2937")
        ),

        "TABLE_HEADER": ParagraphStyle(
            "TABLE_HEADER",

            fontName=bold_font(),

            fontSize=9.2,

            leading=12.5,

            textColor=white
        )
    }


# ============================================================
# TEXT
# ============================================================

def escape_xml(text):

    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def inline_markup(text):

    text = escape_xml(text)

    # Bold
    text = re.sub(
        r"\*\*(.+?)\*\*",
        r"<b>\1</b>",
        text
    )

    text = re.sub(
        r"__(.+?)__",
        r"<b>\1</b>",
        text
    )

    # Italic
    text = re.sub(
        r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)",
        r"<i>\1</i>",
        text
    )

    text = re.sub(
        r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)",
        r"<i>\1</i>",
        text
    )

    return text


# ============================================================
# SMART HEADING DETECTION
# ============================================================

def is_heading(line, next_line=""):

    text = line.strip()

    if not text:
        return False

    if len(text) > 85:
        return False

    # Don't classify sentences as headings
    if text.endswith(
        (
            ".",
            "!",
            "?",
            "।",
            ",",
            ";"
        )
    ):
        return False

    words = text.split()

    if len(words) > 12:
        return False

    # Common heading words
    common = {

        "introduction",
        "overview",
        "summary",
        "conclusion",
        "background",
        "objectives",
        "features",
        "benefits",
        "advantages",
        "disadvantages",
        "examples",

        "परिचय",
        "सारांश",
        "निष्कर्ष",
        "उद्देश",
        "फायदे",
        "तोटे",
        "उदाहरणे",
        "महत्त्वाचे"
    }

    if words[0].lower() in common:
        return True

    # ALL CAPS
    letters = [
        c for c in text
        if c.isalpha()
    ]

    if letters:

        if all(
            c.isupper()
            for c in letters
        ):

            return True

    # Short line followed by long paragraph
    if next_line:

        nxt = next_line.strip()

        if (
            len(text) <= 60
            and len(nxt) >= 45
            and len(nxt) > len(text)
        ):

            return True

    return False


def heading_level(text):

    words = text.split()

    if len(words) <= 6:
        return "H1"

    if len(words) <= 12:
        return "H2"

    return "H3"


# ============================================================
# IMAGES
# ============================================================

def build_image(image_data, max_width):

    try:

        data = image_data.get(
            "data",
            ""
        )

        if data.startswith("data:"):

            data = data.split(
                ",",
                1
            )[1]

        raw = base64.b64decode(
            data
        )

        image = PILImage.open(
            io.BytesIO(raw)
        )

        if image.mode == "RGBA":

            background = PILImage.new(
                "RGB",
                image.size,
                "white"
            )

            background.paste(
                image,
                mask=image.getchannel("A")
            )

            image = background

        elif image.mode != "RGB":

            image = image.convert(
                "RGB"
            )

        width, height = image.size

        if width <= 0 or height <= 0:
            return None

        scale = min(
            max_width / width,
            390 / height,
            1
        )

        new_width = width * scale
        new_height = height * scale

        output = io.BytesIO()

        image.save(
            output,
            "JPEG",
            quality=88
        )

        output.seek(0)

        return KeepTogether(
            [
                Spacer(1, 6),

                RLImage(
                    output,
                    width=new_width,
                    height=new_height
                ),

                Spacer(1, 10)
            ]
        )

    except Exception as error:

        logger.error(
            "Image error: %s",
            error
        )

        return None


# ============================================================
# TABLE
# ============================================================

def table_separator(line):

    cells = [
        x.strip()
        for x in line
        .strip("|")
        .split("|")
    ]

    return (
        len(cells) >= 2
        and all(
            re.fullmatch(
                r":?-{3,}:?",
                cell
            )
            for cell in cells
        )
    )


def build_table(rows, styles, theme_hex, width):

    data = []

    header = [
        Paragraph(
            inline_markup(cell),
            styles["TABLE_HEADER"]
        )
        for cell in rows[0]
    ]

    data.append(header)

    for row in rows[1:]:

        data.append(
            [
                Paragraph(
                    inline_markup(cell),
                    styles["TABLE"]
                )
                for cell in row
            ]
        )

    columns = len(rows[0])

    table = Table(
        data,

        colWidths=[
            width / columns
            for _ in range(columns)
        ],

        repeatRows=1
    )

    table.setStyle(
        TableStyle(
            [

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    HexColor(theme_hex)
                ),

                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    white
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.45,
                    HexColor("#CBD5E1")
                ),

                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        white,
                        HexColor("#F8FAFC")
                    ]
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5
                )
            ]
        )
    )

    return table


# ============================================================
# STORY
# ============================================================

def build_story(
    text,
    images,
    styles,
    theme_hex,
    content_width,
    title
):

    lines = (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .split("\n")
    )

    story = []

    bullets = []
    numbers = []
    paragraph = []

    def flush_paragraph():

        nonlocal paragraph

        if paragraph:

            combined = " ".join(
                line.strip()
                for line in paragraph
            )

            if combined:

                story.append(
                    Paragraph(
                        inline_markup(
                            combined
                        ),
                        styles["BODY"]
                    )
                )

            paragraph = []

    def flush_bullets():

        nonlocal bullets

        if not bullets:
            return

        items = [
            ListItem(
                Paragraph(
                    item,
                    styles["BULLET"]
                )
            )
            for item in bullets
        ]

        story.append(
            ListFlowable(
                items,
                bulletType="bullet",
                leftIndent=18,
                bulletColor=HexColor(
                    theme_hex
                )
            )
        )

        story.append(
            Spacer(1, 7)
        )

        bullets = []

    def flush_numbers():

        nonlocal numbers

        if not numbers:
            return

        items = [
            ListItem(
                Paragraph(
                    item,
                    styles["BULLET"]
                )
            )
            for item in numbers
        ]

        story.append(
            ListFlowable(
                items,
                bulletType="1",
                leftIndent=18,
                bulletColor=HexColor(
                    theme_hex
                )
            )
        )

        story.append(
            Spacer(1, 7)
        )

        numbers = []

    def flush_all():

        flush_paragraph()
        flush_bullets()
        flush_numbers()

    # Main title
    if title:

        first = next(
            (
                x.strip()
                for x in lines
                if x.strip()
            ),
            ""
        )

        if first != title:

            story.append(
                Paragraph(
                    inline_markup(title),
                    styles["TITLE"]
                )
            )

            story.append(
                HRFlowable(
                    width="100%",
                    thickness=2.5,
                    color=HexColor(
                        theme_hex
                    ),
                    spaceAfter=12
                )
            )

    i = 0

    while i < len(lines):

        raw = lines[i]

        line = raw.strip()

        next_line = (
            lines[i + 1].strip()
            if i + 1 < len(lines)
            else ""
        )

        # Empty
        if not line:

            flush_all()

            story.append(
                Spacer(1, 3)
            )

            i += 1
            continue

        # Image
        image_match = re.fullmatch(
            r"\[\[image:(\d+)\]\]",
            line
        )

        if image_match:

            flush_all()

            image = images.get(
                image_match.group(1)
            )

            if image:

                flow = build_image(
                    image,
                    content_width
                )

                if flow:
                    story.append(flow)

            i += 1
            continue

        # Markdown H3
        if line.startswith("### "):

            flush_all()

            story.append(
                Paragraph(
                    inline_markup(
                        line[4:]
                    ),
                    styles["H3"]
                )
            )

            i += 1
            continue

        # Markdown H2
        if line.startswith("## "):

            flush_all()

            story.append(
                Paragraph(
                    inline_markup(
                        line[3:]
                    ),
                    styles["H2"]
                )
            )

            story.append(
                HRFlowable(
                    width=75,
                    thickness=2.2,
                    color=HexColor(
                        theme_hex
                    ),
                    spaceAfter=7
                )
            )

            i += 1
            continue

        # Markdown H1
        if line.startswith("# "):

            flush_all()

            story.append(
                Paragraph(
                    inline_markup(
                        line[2:]
                    ),
                    styles["H1"]
                )
            )

            story.append(
                HRFlowable(
                    width=110,
                    thickness=3,
                    color=HexColor(
                        theme_hex
                    ),
                    spaceAfter=8
                )
            )

            i += 1
            continue

        # Quote
        if line.startswith(">"):

            flush_all()

            story.append(
                Paragraph(
                    inline_markup(
                        line[1:].strip()
                    ),
                    styles["QUOTE"]
                )
            )

            i += 1
            continue

        # Bullet
        bullet = re.match(
            r"^[-*•]\s+(.*)",
            line
        )

        if bullet:

            flush_paragraph()
            flush_numbers()

            bullets.append(
                inline_markup(
                    bullet.group(1)
                )
            )

            i += 1
            continue

        # Number
        number = re.match(
            r"^\d+[.)]\s+(.*)",
            line
        )

        if number:

            flush_paragraph()
            flush_bullets()

            numbers.append(
                inline_markup(
                    number.group(1)
                )
            )

            i += 1
            continue

        # Automatic heading
        if is_heading(
            line,
            next_line
        ):

            flush_all()

            level = heading_level(
                line
            )

            story.append(
                Paragraph(
                    inline_markup(line),
                    styles[level]
                )
            )

            if level in (
                "H1",
                "H2"
            ):

                story.append(
                    HRFlowable(
                        width=105
                        if level == "H1"
                        else 75,

                        thickness=2.8
                        if level == "H1"
                        else 2.2,

                        color=HexColor(
                            theme_hex
                        ),

                        spaceAfter=7
                    )
                )

            i += 1
            continue

        # Normal text
        flush_bullets()
        flush_numbers()

        paragraph.append(line)

        i += 1

    flush_all()

    return story


# ============================================================
# PDF
# ============================================================

def create_pdf(
    text,
    title,
    theme_hex,
    images
):

    ensure_fonts()

    buffer = io.BytesIO()

    margin_left = 20 * mm
    margin_right = 20 * mm
    margin_top = 24 * mm
    margin_bottom = 23 * mm

    document = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        leftMargin=margin_left,

        rightMargin=margin_right,

        topMargin=margin_top,

        bottomMargin=margin_bottom,

        title=title,

        author="Text to PDF Generator",

        subject="Generated document",

        creator="Flask + ReportLab"
    )

    page_width = A4[0]

    content_width = (
        page_width
        - margin_left
        - margin_right
    )

    styles = build_styles(
        theme_hex
    )

    story = build_story(

        text,

        images,

        styles,

        theme_hex,

        content_width,

        title
    )

    if not story:

        story = [
            Paragraph(
                "No content provided.",
                styles["BODY"]
            )
        ]

    canvas_class = make_canvas(
        theme_hex,
        title
    )

    document.build(
        story,
        canvasmaker=canvas_class
    )

    buffer.seek(0)

    return buffer


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():

    ensure_fonts()

    return jsonify({

        "status": "ok",

        "service":
            "Text to PDF Generator",

        "fonts_ready":
            _fonts_ready,

        "message":
            "Backend is running. Use POST /generate-pdf."
    })


@app.route("/health")
def health():

    ensure_fonts()

    return jsonify({

        "status": "ok",

        "fonts_ready":
            _fonts_ready
    })


@app.route(
    "/generate-pdf",
    methods=["POST"]
)
def generate_pdf():

    try:

        data = request.get_json(
            force=True,
            silent=True
        )

        if not isinstance(
            data,
            dict
        ):

            return jsonify({
                "error":
                    "Invalid JSON body"
            }), 400

        text = str(
            data.get(
                "text",
                ""
            )
        )

        if not text.strip():

            return jsonify({
                "error":
                    "Text content is required"
            }), 400

        if len(text) > 2_000_000:

            return jsonify({
                "error":
                    "Text is too large"
            }), 413

        # Support old and new frontend names
        theme_hex = clean_theme(

            data.get(
                "themeColor"
            )

            or data.get(
                "color"
            )

            or data.get(
                "accentColor"
            )

            or data.get(
                "theme"
            )

            or "#2563EB"
        )

        title = str(

            data.get(
                "title"
            )

            or data.get(
                "documentTitle"
            )

            or "Document"

        ).strip()[:160]

        # Images
        image_list = data.get(
            "images",
            []
        )

        image_map = {}

        if isinstance(
            image_list,
            list
        ):

            for image in image_list:

                if not isinstance(
                    image,
                    dict
                ):
                    continue

                index = str(
                    image.get(
                        "index",
                        ""
                    )
                ).strip()

                if index:

                    image_map[index] = image

        ensure_fonts()

        pdf = create_pdf(

            text=text,

            title=title,

            theme_hex=theme_hex,

            images=image_map
        )

        safe_name = re.sub(

            r"[^A-Za-z0-9_\-\u0900-\u097F]+",

            "_",

            title

        ).strip("_") or "document"

        return send_file(

            pdf,

            mimetype="application/pdf",

            as_attachment=True,

            download_name=
                f"{safe_name}.pdf"
        )

    except Exception as error:

        logger.exception(
            "PDF generation failed"
        )

        return jsonify({

            "error":
                f"PDF generation failed: {error}"
        }), 500


# ============================================================
# ERRORS
# ============================================================

@app.errorhandler(413)
def too_large(error):

    return jsonify({

        "error":
            "File/request is too large."
    }), 413


@app.errorhandler(404)
def not_found(error):

    return jsonify({

        "error":
            "Endpoint not found."
    }), 404


@app.errorhandler(500)
def server_error(error):

    return jsonify({

        "error":
            "Internal server error."
    }), 500


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    ensure_fonts()

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(

        host="0.0.0.0",

        port=port,

        debug=False,

        threaded=True
    )
