"""
Internal document structure model.

This layer is intentionally independent of both Gemini and the PDF engine.
Both the Gemini-backed structurer and the deterministic fallback parser
produce a Document made of these dataclasses. The renderer only ever
consumes a Document -- it never sees raw text or JSON. This means the
rendering engine (currently WeasyPrint) can be swapped later without
touching the structuring/parsing layer at all.
"""

from dataclasses import dataclass, field
from typing import List, Optional

ALLOWED_BLOCK_TYPES = {
    "heading",
    "paragraph",
    "bullet_list",
    "numbered_list",
    "quote",
    "table",
    "image",
    "divider",
}


@dataclass
class Block:
    type: str
    text: str = ""
    level: int = 1  # only meaningful for type == "heading" (1, 2, or 3)
    items: List[str] = field(default_factory=list)  # bullet_list / numbered_list
    rows: List[List[str]] = field(default_factory=list)  # table
    image_index: Optional[str] = None  # image

    def is_valid(self) -> bool:
        if self.type not in ALLOWED_BLOCK_TYPES:
            return False
        if self.type == "heading":
            return isinstance(self.text, str) and self.level in (1, 2, 3)
        if self.type in ("paragraph", "quote"):
            return isinstance(self.text, str) and len(self.text.strip()) > 0
        if self.type in ("bullet_list", "numbered_list"):
            return isinstance(self.items, list) and len(self.items) > 0
        if self.type == "table":
            return isinstance(self.rows, list) and len(self.rows) > 0
        if self.type == "image":
            return self.image_index is not None
        if self.type == "divider":
            return True
        return False


@dataclass
class Document:
    title: str = ""
    blocks: List[Block] = field(default_factory=list)
    source: str = "fallback"  # "gemini" | "fallback" | "mixed"
    warnings: List[str] = field(default_factory=list)

    def block_count(self) -> int:
        return len(self.blocks)
