"""
Deterministic fallback structure parser.

CRITICAL FIX vs. the previous version: this parser now takes an explicit
`allow_title` flag. When False, the first line of the given text is NEVER
consumed as a document title -- it is treated as ordinary content (a
heading or paragraph like anything else).

Why this matters: when a large document is split into segments/chunks and
Gemini fails on some later chunk, that chunk gets parsed by this function
as a fallback. Without `allow_title=False`, this parser would treat the
first line of THAT CHUNK as a title, silently replacing (and losing) any
document title already established, and the chunk's actual first line
(e.g. "Rural Solar Solutions") would disappear from the rendered document
entirely instead of appearing as a heading. That was a real data-loss bug.

The rule enforced here: only the caller processing the very first text
segment of the very first chunk of the ENTIRE document may pass
allow_title=True. Every other call -- every fallback chunk, every
non-first segment -- must pass allow_title=False. This function has no
way to know its position in the document; that responsibility belongs to
the caller (app.py), which is exactly why the flag is explicit rather
than inferred here.

Also handles (unchanged from before):
- Markdown-style: #, ##, ### headings; -, * bullets; 1./1) numbered lists;
  > quotes; --- dividers; **bold**/*italic* inline markup.
- Plain text without Markdown: infers headings from short, blank-line-
  isolated standalone lines (language-agnostic -- works identically for
  English, Hindi, and Marathi since it depends on line shape, not grammar).
- [[image:N]] manual image placement markers.

This parser never truncates, reorders, or drops any non-blank line of
input -- every line becomes part of exactly one block.
"""

import re
from typing import List, Optional

from document_model import Block, Document

_BULLET_RE = re.compile(r'^[-*]\s+(.*)')
_NUMBERED_RE = re.compile(r'^\d+[.)]\s+(.*)')
_QUOTE_RE = re.compile(r'^>\s?(.*)')
_IMAGE_RE = re.compile(r'^\[\[image:([A-Za-z0-9_-]+)\]\]$')
_HEADING_RE = re.compile(r'^(#{1,3})\s+(.*)')
_TABLE_ROW_RE = re.compile(r'^\|(.+)\|$')


def _looks_like_heading(line: str, prev_blank: bool) -> bool:
    """
    Heuristic for plain text without Markdown: a short, standalone line
    (no terminal sentence punctuation) isolated by a preceding blank line
    is very likely a heading rather than a full sentence. Depends only on
    line length/shape and blank-line isolation, so it behaves identically
    for Devanagari text as for Latin text.
    """
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > 80:
        return False
    if len(stripped.split()) > 12:
        return False
    if stripped[-1] in '.,;:':
        return False
    return prev_blank


def _parse_table_row(line: str) -> List[str]:
    inner = line.strip()[1:-1]
    return [cell.strip() for cell in inner.split('|')]


def parse_fallback(text: str, allow_title: bool = True) -> Document:
    """
    Parse raw (already Unicode-normalized) text into a Document using only
    deterministic rules -- no external calls, fully reproducible.

    allow_title:
        True  -- this text is the true beginning of the whole document;
                 a title MAY be extracted from its first line/heading.
        False -- this text is a fallback chunk, or any segment that is not
                 the document's very first text segment; NEVER extract a
                 title from it. Every line still becomes a block -- title
                 detection being disabled must never cause content loss.
    """
    lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')

    doc = Document(title="", source="fallback")
    blocks: List[Block] = []

    bullet_buf: List[str] = []
    numbered_buf: List[str] = []
    table_buf: List[List[str]] = []
    # If title extraction is disallowed, treat it as already "used up" so
    # none of the title-consuming branches below ever fire.
    title_assigned = not allow_title

    def flush_bullets():
        nonlocal bullet_buf
        if bullet_buf:
            blocks.append(Block(type="bullet_list", items=bullet_buf))
            bullet_buf = []

    def flush_numbered():
        nonlocal numbered_buf
        if numbered_buf:
            blocks.append(Block(type="numbered_list", items=numbered_buf))
            numbered_buf = []

    def flush_table():
        nonlocal table_buf
        if table_buf:
            blocks.append(Block(type="table", rows=table_buf))
            table_buf = []

    def flush_all():
        flush_bullets()
        flush_numbered()
        flush_table()

    prev_blank = True  # start-of-text counts as "preceded by blank"

    for i, raw_line in enumerate(lines):
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_all()
            prev_blank = True
            continue

        img_match = _IMAGE_RE.match(stripped)
        if img_match:
            flush_all()
            blocks.append(Block(type="image", image_index=img_match.group(1)))
            prev_blank = False
            continue

        if stripped in ('---', '***', '___'):
            flush_all()
            blocks.append(Block(type="divider"))
            prev_blank = False
            continue

        heading_match = _HEADING_RE.match(stripped)
        if heading_match:
            flush_all()
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            if not title_assigned and level == 1:
                doc.title = heading_text
                title_assigned = True
            else:
                blocks.append(Block(type="heading", level=level, text=heading_text))
            prev_blank = False
            continue

        bullet_match = _BULLET_RE.match(stripped)
        if bullet_match:
            flush_numbered()
            flush_table()
            bullet_buf.append(bullet_match.group(1).strip())
            prev_blank = False
            continue

        numbered_match = _NUMBERED_RE.match(stripped)
        if numbered_match:
            flush_bullets()
            flush_table()
            numbered_buf.append(numbered_match.group(1).strip())
            prev_blank = False
            continue

        quote_match = _QUOTE_RE.match(stripped)
        if quote_match:
            flush_all()
            blocks.append(Block(type="quote", text=quote_match.group(1).strip()))
            prev_blank = False
            continue

        table_match = _TABLE_ROW_RE.match(stripped)
        if table_match and '---' not in stripped.replace(' ', ''):
            flush_bullets()
            flush_numbered()
            table_buf.append(_parse_table_row(stripped))
            prev_blank = False
            continue
        elif table_match:
            # markdown table separator row (|---|---|) -- ignore, keep buffering
            prev_blank = False
            continue

        # No Markdown marker matched: plain-text heading heuristic, else paragraph.
        if not title_assigned and i == 0:
            doc.title = stripped
            title_assigned = True
            prev_blank = False
            continue

        if _looks_like_heading(stripped, prev_blank):
            flush_all()
            blocks.append(Block(type="heading", level=2, text=stripped))
            prev_blank = False
            continue

        flush_all()
        blocks.append(Block(type="paragraph", text=stripped))
        prev_blank = False

    flush_all()
    doc.blocks = [b for b in blocks if b.is_valid()]
    return doc
