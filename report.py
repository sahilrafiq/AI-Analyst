"""
report.py - turn a Markdown report into Word (.docx) and HTML, and check its numbers.

Supported Markdown: # headings, paragraphs, **bold**, *italic*, `code`,
- bullet lists, 1. numbered lists, | tables |, and images: ![caption](chart.png)

    from report import to_docx, to_html, unsupported_numbers
    to_docx(markdown_text, "report.docx", title="Sales report", base_dir="workspace")
    to_html(markdown_text, "report.html", title="Sales report", base_dir="workspace")
    unsupported_numbers(markdown_text, analysis_output_text)   # numbers the analysis never produced

Needs python-docx for Word output:  pip install python-docx
"""
import base64
import html as _html
import mimetypes
import re
from pathlib import Path

# ------------------------------------------------------------------- parsing
_HEADING = re.compile(r"^(#{1,4})\s+(.*)")
_IMAGE = re.compile(r"^!\[(.*?)\]\((.*?)\)\s*$")
_BULLET = re.compile(r"^\s*[-*]\s+(.*)")
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+(.*)")
_ESC = re.compile(r"\\([\\`*_{}\[\]()#+.!|-])")  # markdown backslash escapes such as Age\_Group
_INLINE = re.compile(r"(\*\*.+?\*\*|\*[^*\s][^*]*?\*|`.+?`)")


def _is_table_sep(line):
    return bool(re.fullmatch(r"[\s|:\-]+", line)) and "-" in line


def _cells(line):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def _is_special(line):
    s = line.strip()
    return bool(_HEADING.match(s) or _IMAGE.match(s) or s.startswith("|") or _BULLET.match(line) or _NUMBERED.match(line))


def parse_markdown(md):
    """Return a list of blocks: ('h', level, text) ('p', text) ('ul', [items]) ('ol', [items])
    ('img', caption, src) ('table', header, rows)."""
    lines = md.replace("\r\n", "\n").split("\n")
    blocks, i = [], 0
    while i < len(lines):
        line = lines[i].rstrip()
        s = line.strip()
        if not s:
            i += 1
            continue
        m = _HEADING.match(s)
        if m:
            blocks.append(("h", len(m.group(1)), m.group(2).strip()))
            i += 1
            continue
        m = _IMAGE.match(s)
        if m:
            blocks.append(("img", m.group(1).strip(), m.group(2).strip()))
            i += 1
            continue
        if s.startswith("|") and i + 1 < len(lines) and _is_table_sep(lines[i + 1].strip()):
            header = _cells(s)
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_cells(lines[i]))
                i += 1
            blocks.append(("table", header, rows))
            continue
        m = _BULLET.match(line)
        if m:
            items = []
            while i < len(lines) and _BULLET.match(lines[i]):
                items.append(_BULLET.match(lines[i]).group(1).strip())
                i += 1
            blocks.append(("ul", items))
            continue
        m = _NUMBERED.match(line)
        if m:
            items = []
            while i < len(lines) and _NUMBERED.match(lines[i]):
                items.append(_NUMBERED.match(lines[i]).group(1).strip())
                i += 1
            blocks.append(("ol", items))
            continue
        para = [s]
        i += 1
        while i < len(lines) and lines[i].strip() and not _is_special(lines[i]):
            para.append(lines[i].strip())
            i += 1
        blocks.append(("p", " ".join(para)))
    return blocks


def _runs(text):
    """Split inline Markdown into (text, bold, italic, code) pieces."""
    out = []
    for part in _INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            out.append((_ESC.sub(r"\1", part[2:-2]), True, False, False))
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            out.append((part[1:-1].replace("\\_", "_"), False, False, True))
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            out.append((_ESC.sub(r"\1", part[1:-1]), False, True, False))
        else:
            out.append((_ESC.sub(r"\1", part), False, False, False))
    return out


def _plain(text):
    return "".join(t for t, *_ in _runs(text))


# ---------------------------------------------------------------------- Word
def to_docx(md, path, title=None, base_dir="."):
    """Build a .docx from Markdown. Returns a list of image files that could not be found."""
    try:
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Inches, Pt
    except ImportError:
        raise ImportError("Word output needs python-docx:  pip install python-docx")

    base = Path(base_dir)
    doc = Document()
    if title:
        doc.add_heading(title, 0)
    missing = []

    def add_runs(par, text):
        for t, bold, italic, code in _runs(text):
            r = par.add_run(t)
            r.bold, r.italic = bold or None, italic or None
            if code:
                r.font.name = "Consolas"
                r.font.size = Pt(9.5)

    for blk in parse_markdown(md):
        kind = blk[0]
        if kind == "h":
            if title and blk[1] == 1 and _plain(blk[2]).strip().lower() == title.strip().lower():
                continue  # don't repeat the title
            doc.add_heading(_plain(blk[2]), level=min(blk[1], 4))
        elif kind == "p":
            add_runs(doc.add_paragraph(), blk[1])
        elif kind in ("ul", "ol"):
            for item in blk[1]:
                add_runs(doc.add_paragraph(style="List Bullet" if kind == "ul" else "List Number"), item)
        elif kind == "img":
            f = base / blk[2]
            if f.exists():
                doc.add_picture(str(f), width=Inches(6.0))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                if blk[1]:
                    cap = doc.add_paragraph()
                    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    r = cap.add_run(blk[1])
                    r.italic = True
                    r.font.size = Pt(9)
            else:
                missing.append(blk[2])
                doc.add_paragraph(f"[image not found: {blk[2]}]")
        elif kind == "table":
            header, rows = blk[1], blk[2]
            ncol = len(header)
            table = doc.add_table(rows=1, cols=ncol)
            table.style = "Light Grid Accent 1"
            for j, h in enumerate(header):
                cell = table.rows[0].cells[j]
                cell.text = ""
                r = cell.paragraphs[0].add_run(_plain(h))
                r.bold = True
            for row in rows:
                cells = table.add_row().cells
                for j in range(ncol):
                    cells[j].text = _plain(row[j]) if j < len(row) else ""
    doc.save(str(path))
    return missing


# ---------------------------------------------------------------------- HTML
_CSS = """body{font-family:Segoe UI,Arial,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;color:#1b1b1b;line-height:1.55}
h1,h2,h3{color:#1f3864}table{border-collapse:collapse;margin:1rem 0;width:100%}th,td{border:1px solid #ccc;padding:6px 10px;text-align:left}
th{background:#1f3864;color:#fff}img{max-width:100%}figure{margin:1.2rem 0;text-align:center}figcaption{font-style:italic;font-size:.9rem;color:#555}
code{background:#f3f3f3;padding:1px 4px;border-radius:3px}"""


def _html_inline(text):
    out = []
    for t, bold, italic, code in _runs(text):
        t = _html.escape(t)
        out.append(f"<strong>{t}</strong>" if bold else f"<em>{t}</em>" if italic else f"<code>{t}</code>" if code else t)
    return "".join(out)


def to_html(md, path, title="Report", base_dir="."):
    """Build one self-contained .html file (charts embedded). Returns a list of missing images."""
    base, missing, body = Path(base_dir), [], [f"<h1>{_html.escape(title)}</h1>"]
    for blk in parse_markdown(md):
        kind = blk[0]
        if kind == "h":
            if blk[1] == 1 and _plain(blk[2]).strip().lower() == title.strip().lower():
                continue
            lvl = min(blk[1] + 1, 4) if blk[1] == 1 else min(blk[1], 4)
            body.append(f"<h{lvl}>{_html_inline(blk[2])}</h{lvl}>")
        elif kind == "p":
            body.append(f"<p>{_html_inline(blk[1])}</p>")
        elif kind in ("ul", "ol"):
            body.append(f"<{kind}>" + "".join(f"<li>{_html_inline(i)}</li>" for i in blk[1]) + f"</{kind}>")
        elif kind == "img":
            f = base / blk[2]
            if f.exists():
                mime = mimetypes.guess_type(f.name)[0] or "image/png"
                data = base64.b64encode(f.read_bytes()).decode()
                cap = f"<figcaption>{_html.escape(blk[1])}</figcaption>" if blk[1] else ""
                body.append(f'<figure><img src="data:{mime};base64,{data}" alt="{_html.escape(blk[1])}">{cap}</figure>')
            else:
                missing.append(blk[2])
                body.append(f"<p><em>[image not found: {_html.escape(blk[2])}]</em></p>")
        elif kind == "table":
            head = "".join(f"<th>{_html_inline(h)}</th>" for h in blk[1])
            rows = "".join("<tr>" + "".join(f"<td>{_html_inline(c)}</td>" for c in r) + "</tr>" for r in blk[2])
            body.append(f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>")
    doc = (
        f'<!doctype html><html><head><meta charset="utf-8"><title>{_html.escape(title)}</title>'
        f"<style>{_CSS}</style></head><body>{''.join(body)}</body></html>"
    )
    Path(path).write_text(doc, encoding="utf-8")
    return missing


# --------------------------------------------------------- number verification
_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?(?![\w])")


def _numbers(text):
    found = []
    for m in _NUM.finditer(text):
        tok = m.group(0)
        raw = tok.rstrip("%").replace(",", "")
        try:
            val = float(raw)
        except ValueError:
            continue
        found.append((tok, val, len(raw.split(".")[1]) if "." in raw else 0, tok.endswith("%")))
    return found


def unsupported_numbers(report_md, source_text, limit=15):
    """Numbers that appear in the report but cannot be found in the analysis output
    (within rounding). Skips small whole numbers and years. These need a human check."""
    clean = re.sub(r"!\[.*?\]\(.*?\)", " ", report_md)  # ignore image paths
    clean = "\n".join(l for l in clean.splitlines() if not _is_table_sep(l.strip()))
    src = [v for _, v, _, _ in _numbers(source_text)]
    cands = src + [v * 100 for v in src if abs(v) <= 1.5]  # fractions shown as percentages
    bad, seen = [], set()
    for tok, v, dec, pct in _numbers(clean):
        if dec == 0 and not pct and (abs(v) < 100 or 1900 <= abs(v) <= 2100):
            continue
        if any(abs(v - y) <= max(0.5 * 10 ** (-dec), 0.005 * abs(y)) for y in cands):
            continue
        if tok not in seen:
            seen.add(tok)
            bad.append(tok)
    return bad[:limit]


_GENERIC = set(
    """the and for with was were has have had are that this from than then into over under per all any not but its their there
    mean median min max std standard deviation average avg total sum count number value values missing unique rows row columns column
    int float object dtype datetime string bool top test statistic stat result results data dataset records record duplicate duplicates
    observations about each across which shows show found""".split()
)


def _words(text):
    ws = re.findall(r"[a-z]{3,}", text.lower().replace("_", " "))
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in ws} - {g[:-1] if g.endswith("s") and len(g) > 3 else g for g in _GENERIC} - _GENERIC


def mislabeled_numbers(report_md, source_text, limit=8):
    """Numbers that DO appear in the analysis output, but only next to different labels than the
    report sentence uses (e.g. report says 'customer age', output shows it beside 'Unit_Cost').
    These are hints to verify, not proof of an error."""
    idx = [(_numbers(l), _words(l), l) for l in source_text.splitlines() if l.strip()]
    clean = re.sub(r"!\[.*?\]\(.*?\)", " ", report_md)
    clean = "\n".join(l for l in clean.splitlines() if not _is_table_sep(l.strip()))
    out, seen = [], set()
    for sent in re.split(r"(?<=[.!?])\s+|\n", clean):
        sw = _words(sent)
        if not sw:
            continue
        for tok, v, dec, pct in _numbers(sent):
            if (dec == 0 and not pct and (abs(v) < 100 or 1900 <= abs(v) <= 2100)) or tok in seen:
                continue
            hits = [(w, l) for nums, w, l in idx if any(abs(v - y) <= max(0.5 * 10 ** (-dec), 0.005 * abs(y)) for _, y, _, _ in nums)]
            if not hits or any((not w) or (w & sw) for w, _ in hits):
                continue  # not in output at all (covered elsewhere) or label matches
            seen.add(tok)
            out.append(f"{tok} in \"{sent.strip()[:80]}\" - in the output it appears next to: {hits[0][1].strip()[:70]}")
    return out[:limit]
