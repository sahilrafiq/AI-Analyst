"""
bench_excel.py - Excel tasks with exact grading.

Builds a messy workbook (report.xlsx: title rows, a Total row, vertically merged region cells,
formulas, a notes sheet) and asks the agent to read it correctly and to write Excel files with
live formulas and native charts. Files the agent creates are opened and inspected, so a pass
means a real formula / real chart exists, not just the right numbers.

Keep agent.py, analyst_tools.py, bench_hard.py and this file in one folder.

    python bench_excel.py
    python bench_excel.py --models gemma4:e4b qwen3.5:4b
Transcripts are saved in bench_excel_work/transcript_*.md
"""
import argparse
import re
import time
from pathlib import Path

import numpy as np

from agent import Agent
from bench_hard import _f, parse_result

WORK = Path("bench_excel_work")

INTRO = "The workbook report.xlsx has three sheets: 'Summary', 'Regional' and 'Notes'. "
TASKS = {
    "xl_read": INTRO + "The sheet 'Summary' has title lines above its table and a Total row at the bottom. Print exactly one line: "
    "RESULT rows=<number of data rows, not counting the Total row> revenue=<total Revenue of the data rows, 2 decimals>",
    "xl_merged": INTRO + "In the sheet 'Regional' the Region name is only filled in on the first row of each block (merged cells), and there is a "
    "Grand total row. Print exactly one line: RESULT north=<total Revenue of North> west=<total Revenue of West>",
    "xl_write": INTRO + "Create summary.xlsx with one sheet named 'Summary' that holds the total Revenue per Region (computed from the data rows of "
    "the 'Summary' sheet of report.xlsx, excluding its Total row; columns named Region and Revenue), a Total row at the bottom that uses live "
    "SUM formulas, and one native Excel bar chart of Revenue by Region. Print exactly one line: RESULT saved=summary.xlsx",
    "xl_formula": INTRO + "Create margin.xlsx with one sheet named 'Data' that holds the table from the 'Summary' sheet of report.xlsx (data rows only, "
    "no Total row) plus a new last column named 'Avg Price' whose cells are live Excel formulas =Revenue/Units for every row (formulas, not pasted "
    "values). Print exactly one line: RESULT saved=margin.xlsx",
}


# ------------------------------------------------------------------ data + truth
def make_data(workdir):
    from openpyxl import Workbook

    Path(workdir).mkdir(exist_ok=True)
    rng = np.random.default_rng(7)
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws["A1"] = "Q3 Sales Report - CONFIDENTIAL"
    ws.merge_cells("A1:E1")
    ws["A2"] = "Prepared by finance; figures in USD"
    ws.append([])
    ws.append(["Region", "Product", "Units", "Price", "Revenue"])
    regions, data = ["North", "South", "East", "West"], []
    for i in range(40):
        u, p = int(rng.integers(1, 20)), round(float(rng.uniform(5, 50)), 2)
        data.append((regions[i % 4], "ABC"[i % 3], u, p, round(u * p, 2)))
        ws.append(list(data[-1]))
    ws.append(["Total", None, "=SUM(C5:C44)", None, "=SUM(E5:E44)"])
    ws2 = wb.create_sheet("Regional")
    ws2.append(["Region", "Month", "Revenue"])
    blocks, r = {"North": [100, 150, 200], "South": [80, 90, 70], "West": [300, 250, 400]}, 2
    for reg, vals in blocks.items():
        for m, v in zip(["Jan", "Feb", "Mar"], vals):
            ws2.append([reg if m == "Jan" else None, m, v])
        ws2.merge_cells(start_row=r, start_column=1, end_row=r + 2, end_column=1)
        r += 3
    ws2.append(["Grand total", None, sum(sum(v) for v in blocks.values())])
    wb.create_sheet("Notes")["A1"] = "just a note"
    wb.save(Path(workdir) / "report.xlsx")
    per_region = {}
    for reg, _, _, _, rev in data:
        per_region[reg] = round(per_region.get(reg, 0) + rev, 2)
    return {"rows": 40, "revenue": round(sum(d[4] for d in data), 2), "north": 450.0, "west": 950.0, "per_region": per_region}


# ----------------------------------------------------------------------- grading
def _letters(n):
    from openpyxl.utils import get_column_letter

    return get_column_letter(n)


def grade_file(task, workdir, truth):
    from openpyxl import load_workbook

    path = Path(workdir) / ("summary.xlsx" if task == "xl_write" else "margin.xlsx")
    if not path.exists():
        return False, f"{path.name} was not created"
    wb = load_workbook(path)
    sheet = "Summary" if task == "xl_write" else "Data"
    if sheet not in wb.sheetnames:
        return False, f"no sheet named '{sheet}' (found {wb.sheetnames})"
    ws = wb[sheet]
    header = [c.value for c in ws[1]]
    if task == "xl_write":
        if "Region" not in header or "Revenue" not in header:
            return False, f"header should contain Region and Revenue, got {header}"
        ri, vi = header.index("Region") + 1, header.index("Revenue") + 1
        got = {ws.cell(r, ri).value: ws.cell(r, vi).value for r in range(2, ws.max_row + 1)}
        for reg, want in truth["per_region"].items():
            if not isinstance(got.get(reg), (int, float)) or abs(got[reg] - want) > 0.01:
                return False, f"Revenue for {reg}: got {got.get(reg)!r}, want {want}"
        sums = [ws.cell(r, vi).value for r in range(2, ws.max_row + 1) if isinstance(ws.cell(r, vi).value, str)]
        if not any(v.upper().startswith("=SUM(") for v in sums):
            return False, "no live =SUM(...) formula found in the Revenue column"
        if len(ws._charts) < 1:
            return False, "no native chart in the sheet"
        return True, f"values ok, live SUM found, {len(ws._charts)} chart(s)"
    # xl_formula
    if header[-1] != "Avg Price" or not {"Units", "Revenue"} <= set(header):
        return False, f"last header should be 'Avg Price' and Units/Revenue must exist, got {header}"
    letters = {h: _letters(i + 1) for i, h in enumerate(header)}
    nrows = ws.max_row - 1
    if nrows != truth["rows"]:
        return False, f"expected {truth['rows']} data rows, found {nrows}"
    for r in range(2, ws.max_row + 1):
        v = ws.cell(r, len(header)).value
        want = f"={letters['Revenue']}{r}/{letters['Units']}{r}"
        if not (isinstance(v, str) and v.replace(" ", "").upper() == want):
            return False, f"row {r}: expected formula {want}, got {v!r}"
    return True, f"{nrows} live formulas =Revenue/Units"


def grade(task, stdout, workdir, truth):
    if task in ("xl_write", "xl_formula"):
        return grade_file(task, workdir, truth)
    r = parse_result(stdout)
    if r is None:
        return False, "no 'RESULT ...' line printed"
    try:
        if task == "xl_read":
            ok = int(_f(r["rows"])) == truth["rows"] and abs(_f(r["revenue"]) - truth["revenue"]) < 0.02
            return ok, f"got {r} | want rows={truth['rows']} revenue={truth['revenue']}"
        ok = abs(_f(r["north"]) - truth["north"]) < 0.01 and abs(_f(r["west"]) - truth["west"]) < 0.01
        return ok, f"got {r} | want north={truth['north']} west={truth['west']}"
    except (KeyError, ValueError) as e:
        return False, f"could not read result ({e!r}): {r}"


# -------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma4:e4b"])
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--no-helpers", action="store_true")
    args = ap.parse_args()

    truth = make_data(WORK)
    print("Planted truth:", {k: v for k, v in truth.items() if k != "per_region"}, "| per region:", truth["per_region"])
    rows = []
    for model in args.models:
        print(f"\n=== {model} (excel tasks, helpers {'off' if args.no_helpers else 'on'}) ===")
        agent = Agent(model, workdir=WORK, num_ctx=args.num_ctx, max_steps=args.max_steps, helpers=not args.no_helpers)
        try:
            for task, prompt in TASKS.items():
                for stray in WORK.iterdir():  # only report.xlsx survives between tasks
                    if stray.name != "report.xlsx" and not stray.name.startswith(("transcript_", "session_")):
                        stray.unlink()
                agent.reset()
                t0 = time.time()
                try:
                    agent.solve(prompt)
                except Exception as e:
                    print(f"  {task:11} FAIL  agent error: {e}")
                    rows.append((model, task, False, 0, 0))
                    continue
                wall = time.time() - t0
                stdout = "\n".join(h["stdout"] for h in agent.history if not h["error"] and not h.get("warning"))
                ok, note = grade(task, stdout, WORK, truth)
                errs = sum(1 for h in agent.history if h["error"])
                print(f"  {task:11} {'PASS' if ok else 'FAIL'}  steps={agent.last_steps} errors={errs} {wall:4.0f}s  {note}")
                rows.append((model, task, ok, agent.last_steps, wall))
        finally:
            speed = agent.eval_tokens / agent.eval_secs if agent.eval_secs else 0
            agent.close()
            print(f"  average speed: {speed:.1f} tok/s")

    print("\n=== SUMMARY (excel tasks) ===")
    print(f"{'model':14} {'passed':8} {'avg steps':10} {'total time':10}")
    for model in args.models:
        rs = [r for r in rows if r[0] == model]
        if rs:
            print(f"{model:14} {sum(r[2] for r in rs)}/{len(rs):<6} {sum(r[3] for r in rs) / len(rs):<10.1f} {sum(r[4] for r in rs):.0f}s")
    print(f"\nTranscripts of every step are in {WORK}\\transcript_*.md")


if __name__ == "__main__":
    main()
