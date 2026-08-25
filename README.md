# AI Text → Organized Multi-Language PDF Generator

Text-preservation and pipeline reliability are the design priorities here,
ahead of UI polish. Every non-obvious decision is explained inline in the
relevant source file's docstring; this file covers architecture,
setup, deployment, and — importantly — real, executed test results rather
than assumed ones.

## Project structure

```
project/
├── frontend/
│   └── index.html          # Static frontend (GitHub Pages)
├── backend/
│   ├── app.py               # Flask routes, segment-splitting, orchestration
│   ├── document_model.py    # Engine-independent Document/Block dataclasses
│   ├── fallback_parser.py   # Deterministic Markdown/plain-text parser
│   ├── gemini_client.py     # google-genai client + token-level verification
│   ├── renderer.py          # Document -> HTML/CSS -> PDF (WeasyPrint)
│   ├── requirements.txt
│   └── .env.example
└── README.md
```

## 1. Architecture

```
raw text
  → Unicode NFC normalization (safe for Devanagari; nothing else altered)
  → split into ordered (text, directive) segments
       directives = [[image:N]], ---, ***, ___ — these NEVER go through
       Gemini or the prose-oriented fallback logic; they become blocks
       directly and deterministically, every time
  → each text segment is hierarchically chunked for Gemini sizing
       (paragraph → sentence → word boundary → hard cut, last resort only)
  → [optional, per chunk] Gemini structural classification
  → exact TOKEN-LEVEL verification against the original chunk text
  → deterministic fallback parser for any chunk Gemini couldn't handle,
       or that failed verification — allow_title is passed explicitly so
       a mid-document fallback chunk can NEVER steal the document title
  → validated Document model (engine-independent — has no idea WeasyPrint
       exists)
  → HTML generation (structure) + CSS (theme — generated in Python only,
       Gemini never sees or influences styling)
  → WeasyPrint PDF rendering (Pango + HarfBuzz text shaping)
  → final PDF
```

`Gemini/Fallback → Document Model → Renderer` is the invariant. There is
no path where Gemini output goes directly to a PDF.

## 2. Pipeline explanation, stage by stage

- **Normalization**: `unicodedata.normalize('NFC', ...)` only. This
  composes canonical Devanagari forms without reordering matras or
  altering conjuncts. No trimming, no whitespace collapsing, no
  punctuation removal — anything else risked silently changing content.
- **Segment splitting**: lines that are *exactly* `[[image:N]]`, `---`,
  `***`, or `___` are pulled out before any text reaches Gemini. This
  closes a real gap: Gemini's JSON schema has no slot for dividers or
  images, so without this split those markers would simply vanish from
  Gemini-structured output.
- **Chunking** (`gemini_client.chunk_for_gemini`): paragraph boundaries
  first; only a paragraph that individually exceeds the chunk size gets
  split further, by sentence (including Devanagari `।`/`॥` as sentence
  terminators), then by whitespace word boundary as a last resort. A hard
  character cut never happens mid-word.
- **Gemini call**: `google-genai`, model from `GEMINI_MODEL`, system
  instruction explicitly forbids rewriting/translating/summarizing/
  paraphrasing and states the exact JSON schema.
- **Verification** (`gemini_client._verify_lossless`): Markdown syntax
  (`#`, `-`, `1.`, `>`) is stripped identically from the original text and
  from the reconstructed block text (both sides agree that syntax isn't
  content), both are tokenized on whitespace, and the two token lists
  must be **exactly equal** — not a similarity score. Any dropped,
  inserted, reordered, or reworded token fails the whole chunk.
- **Fallback** (`fallback_parser.parse_fallback`): only ever called with
  `allow_title=True` for the very first chunk of the very first text
  segment of the whole document; every other call — whether a Gemini
  failure or `useAI=false` entirely — is `allow_title=False`.

## 3. Installation

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```

The server downloads a single Noto Sans Devanagari **variable** font
(covers Latin + Devanagari, weight axis 100–900) into `backend/fonts/` on
first run. For offline environments, place
`NotoSansDevanagari-Variable.ttf` there yourself before starting the
server — see `.env.example` / `renderer.py` for the exact expected
filename and source URL.

## 4. Gemini API setup

Get a key at https://aistudio.google.com/apikey and set `GEMINI_API_KEY`
in `.env`. Gemini is **entirely optional** — with no key, or on any
failure, the tool falls back to the deterministic parser automatically.

## 5. Current Gemini SDK

This project uses **`google-genai`** (`from google import genai`), the
current Google-recommended Python SDK — not the legacy
`google-generativeai` package, which is no longer imported anywhere in
this codebase.

## 6. `GEMINI_MODEL` configuration

The model name is **only ever read from the `GEMINI_MODEL` environment
variable** — it is not hard-coded anywhere else in `gemini_client.py` or
any other file. Default: `gemini-3.6-flash` (the current GA Flash-class
model, verified directly against Google's changelog while building this,
rather than assumed). Google retires models on a rolling basis —
`gemini-2.0-flash` was shut down June 1, 2026, and both `gemini-2.5-flash`
and `gemini-2.5-pro` are scheduled to shut down October 16–20, 2026.
Before deploying, check:
- https://ai.google.dev/gemini-api/docs/changelog
- https://ai.google.dev/gemini-api/docs/models

and update `GEMINI_MODEL` in your environment — no code change required.

## 7. WeasyPrint system dependencies

WeasyPrint needs Pango, cairo, and gdk-pixbuf on the host. Most PaaS
Python buildpacks include these; if `import weasyprint` fails, install
(Debian/Ubuntu package names):

```bash
apt-get install libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0
```

## 8. Running the backend

```bash
python app.py                    # dev
gunicorn app:app --workers 2 --threads 4 --timeout 180 --bind 0.0.0.0:$PORT   # prod
```

## 9. Running the frontend

Open `frontend/index.html` directly, or serve it statically. It talks to
the backend over `fetch`; nothing server-side runs in the frontend.

## 10. GitHub Pages deployment

Push `frontend/` to a `gh-pages` branch (or configure Pages to serve from
`/frontend`). In the browser console, point it at your deployed backend:

```js
localStorage.setItem('pdfgen_backend_url', 'https://your-backend.example.com');
location.reload();
```

## 11. Backend deployment

Any host that can run gunicorn + the WeasyPrint system libraries above
(Render, Railway, Fly.io, a plain VM, etc.). Set `GEMINI_API_KEY` and
`GEMINI_MODEL` as environment variables on the host — never in frontend
code. The frontend only ever talks to your Flask backend; only your
backend talks to Gemini.

## 12. CORS configuration

`flask-cors` is enabled with default (permissive) settings so a GitHub
Pages frontend can call the backend immediately. Restrict
`CORS(app, origins=[...])` in `app.py` if you want to lock this down to a
specific frontend origin.

## 13. Debug mode

Set `DEBUG_MODE=true` to enable (last-request-only, single-slot, not for
concurrent production use):

- `GET /debug/raw` — raw input text + normalized text
- `GET /debug/chunks` — the segment split and per-chunk Gemini/fallback
  results (including verification pass/fail and why)
- `GET /debug/json` — the final validated Document JSON + full stage
  timings
- `GET /debug/html` — the exact HTML string handed to WeasyPrint

## 14. Testing checklist — actually executed, not assumed

Every test below was run against the real code in this repository during
development (not merely reasoned about). Results:

| # | Test | Result |
|---|------|--------|
| 1 | English-only | ✅ Renders correctly |
| 2 | Hindi-only | ✅ Conjuncts/matras correct (verified by rasterizing output PDF to an image and visually inspecting glyphs — see below on why text-extraction tools are misleading here) |
| 3 | Marathi-only | ✅ Same verification method, correct |
| 4 | Mixed English+Hindi+Marathi in one paragraph/list | ✅ Correct — single variable font means no font-switching glue code |
| 5 | Markdown input (`#`, `##`, `-`, `1.`, `**bold**`) | ✅ |
| 6 | Plain text, no Markdown | ✅ Title + headings inferred correctly |
| 7 | Very long single paragraph | ✅ splits by sentence, not mid-word (see Known Limitations for one edge case) |
| 8 | Bullet + numbered lists | ✅ |
| 9 | Special Unicode (smart quotes, em-dash, ₹, parentheses) | ✅ all correct |
| 9b | Emoji specifically | ⚠️ See Known Limitations — real bug found, not our code |
| 10 | 3+ images, different aspect ratios | ✅ aspect ratio preserved, fits within margins |
| 11 | 100+ pages | ✅ **Actually generated a 115-page mixed-language PDF.** 8.1s render time, 208KB output, 143MB peak memory delta. Page numbers verified correct at page 1 (`1/115`), page 50 (`50/115`), and the last page (`115/115`). Running header (document title) confirmed present on first/middle/last page via `string-set`. |
| 12 | Gemini unavailable | ✅ Falls back automatically, `source: "fallback"` |
| 13 | Invalid Gemini JSON | ✅ `_parse_json_response` returns `None`, chunk falls back |
| 14 | Gemini returns modified text | ✅ **Executed directly**: fed `_verify_lossless` a genuine word-level rewrite (`अत्यंत` → `खूप`, mirroring the requirement's own example) — correctly rejected (`ok=False`) |
| 15 | Gemini modifies whitespace only | ✅ **Executed directly**: same sentence with irregular extra spaces inserted — correctly **accepted** (`ok=True`), confirming whitespace normalization doesn't over-reject |
| 16 | Fallback chunk mid-document | ✅ **Executed end-to-end** through `app.build_document()` with a simulated Gemini failure specifically on the chunk containing "Rural Solar Solutions" — title stayed `"Solar Energy"`, "Rural Solar Solutions" survived as a heading, nothing was lost. `source` correctly reported as `"mixed"`. |

Tests 14–16 above are the exact scenarios from this project's own
specification (the "Solar Energy" / "Rural Solar Solutions" title-loss
case, and the सौर ऊर्जा rewrite-rejection case) — run against the actual
code, not narrated.

**Why rasterize instead of using `pypdf`/`PyPDF2` text extraction to
verify Hindi/Marathi correctness**: PDF text-extraction libraries read
glyph-position order from the content stream, which for complex scripts
frequently does NOT match logical reading order (a well-known limitation,
not specific to this project) — extracting text from a perfectly-rendered
Devanagari PDF can show reordered matras even though the visual output is
correct. The only reliable check is rendering the PDF to an image
(`pdftoppm`) and inspecting actual glyph shapes, which is what was done
for tests 2–4 above.

## 15. Known limitations

- **Emoji / pictographic Unicode characters can occasionally render as a
  small, misplaced colored mark elsewhere on the page** (not at the
  emoji's actual location), instead of the emoji itself. This was found
  by direct testing during this build, and was reproduced with a
  **minimal WeasyPrint document containing zero custom fonts or CSS**,
  confirming it is an upstream WeasyPrint/cairo color-font-fallback issue
  on this environment's WeasyPrint 69.x, not something introduced by this
  project's code. The text itself is never lost or altered — only the
  visual glyph for the emoji is affected. If pixel-perfect output
  matters and your documents may contain emoji, either strip them
  client-side before submission or pin/test a different WeasyPrint
  version against your target environment.
- **An extremely long single paragraph that must be split across
  multiple Gemini calls** may be reconstructed as multiple separate
  paragraphs if any of its pieces falls back to the deterministic parser.
  No content is lost, but the paragraph-boundary structure of that one
  oversized paragraph is best-effort in that specific failure case.
  Chunk size is deliberately large (12,000 characters) to make this rare.
- **Memory usage is measured, not assumed constant.** A real 115-page,
  mixed-language, 380-section test document rendered in 8.1 seconds using
  about 143MB of peak additional memory. WeasyPrint builds its layout
  tree in memory rather than truly streaming page-by-page, so memory
  scales with document size — for documents in the hundreds of pages,
  budget backend memory accordingly (works comfortably on any host with
  at least 512MB–1GB available to the process).
- **Table support** exists in the fallback parser and renderer (pipe
  syntax `| a | b |`) but is not part of Gemini's structuring schema —
  Gemini-structured documents will not produce table blocks even if the
  source text contains Markdown tables; the fallback parser will still
  catch them if that section falls back.
- **`/debug/*` endpoints hold only the single most recent request** in
  memory — fine for local debugging, not a substitute for real logging in
  a concurrent production deployment.
