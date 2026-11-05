"""
benchmark.py - test local Ollama models on real data-analyst tasks.

For each model it asks for Python code for 3 tasks, RUNS that code, and checks
the output against the correct answer (computed here with pandas/scipy).
It also records speed (tokens/second).

Usage (from the folder where you saved this file):
    python benchmark.py
    python benchmark.py --models qwen3.5:4b gemma4:e4b
    python benchmark.py --num-ctx 8192

Needs:  pip install pandas numpy scipy matplotlib openpyxl
Needs Ollama running (the tray app is enough).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
WORK = Path("bench_work")
GEN = WORK / "generated"

SYSTEM = (
    "You are a Python data analysis assistant. "
    "Reply with exactly one python code block and nothing else."
)

DATA_NOTE = (
    "The file sales.csv is in the current folder with columns: "
    "date (YYYY-MM-DD), product, units, price. Revenue = units * price. "
)

TASKS = {
    "chart": DATA_NOTE
    + "For each calendar month find the top 5 products by total revenue and draw ONE "
    "grouped bar chart of them (x axis = month). Save it as chart1.png. "
    "Do not call plt.show().",
    "stats": DATA_NOTE
    + "Compare the 'units' column of product A rows versus product B rows using "
    "Welch's t-test (scipy, equal_var=False, A first). Also compute Cohen's d as "
    "(mean_A - mean_B) / pooled_sd, where pooled_sd uses sample variances (ddof=1). "
    "Print exactly one line in this format: RESULT t=<t> p=<p> d=<d> "
    "(each number rounded to 4 decimals).",
    "excel": DATA_NOTE
    + "Compute total revenue per month, with month formatted as YYYY-MM. Save it to "
    "summary.xlsx on a sheet named Monthly with columns 'month' and 'revenue' "
    "(revenue rounded to 2 decimals), one row per month, sorted by month. Use pandas.",
}


# ----------------------------------------------------------------- data + truth
def check_imports():
    missing = []
    for mod in ["pandas", "numpy", "scipy", "matplotlib", "openpyxl"]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print("Missing packages:", ", ".join(missing))
        print("Install them with:  pip install " + " ".join(missing))
        sys.exit(1)


def make_data():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 400
    days = rng.integers(0, 365, n)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime("2025-01-01") + pd.to_timedelta(days, unit="D"),
            "product": rng.choice(list("ABCDEFGH"), n),
            "units": rng.integers(1, 20, n),
            "price": rng.uniform(5, 50, n).round(2),
        }
    ).sort_values("date")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.to_csv(WORK / "sales.csv", index=False)
    return df


def compute_truth(df):
    import numpy as np
    from scipy import stats

    a = df.loc[df["product"] == "A", "units"].to_numpy(float)
    b = df.loc[df["product"] == "B", "units"].to_numpy(float)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    d = (a.mean() - b.mean()) / sp
    monthly = (
        df.assign(revenue=df["units"] * df["price"], month=df["date"].str[:7])
        .groupby("month")["revenue"]
        .sum()
        .round(2)
    )
    return {"t": float(t), "p": float(p), "d": float(d)}, monthly


# ---------------------------------------------------------------------- checks
def check_chart(stdout, truth):
    f = WORK / "chart1.png"
    if not f.exists():
        return False, "chart1.png was not created"
    if f.stat().st_size < 5000:
        return False, "chart1.png is suspiciously small (empty chart?)"
    return True, "chart saved (open bench_work/chart1.png to judge it by eye)"


def check_stats(stdout, truth):
    m = re.search(
        r"RESULT\s+t=\s*(-?[\d.]+(?:e-?\d+)?)\s+p=\s*(-?[\d.]+(?:e-?\d+)?)\s+d=\s*(-?[\d.]+(?:e-?\d+)?)",
        stdout,
        re.I,
    )
    if not m:
        return False, "no 'RESULT t=.. p=.. d=..' line in output"
    t, p, d = (float(x) for x in m.groups())
    want = truth["stats"]
    ok = abs(t - want["t"]) < 0.001 and abs(p - want["p"]) < 0.0005 and abs(d - want["d"]) < 0.001
    detail = f"got t={t:.3f} p={p:.4f} d={d:.3f} | want t={want['t']:.3f} p={want['p']:.4f} d={want['d']:.3f}"
    return ok, detail


def check_excel(stdout, truth):
    import pandas as pd

    f = WORK / "summary.xlsx"
    if not f.exists():
        return False, "summary.xlsx was not created"
    try:
        got = pd.read_excel(f, sheet_name="Monthly")
    except Exception as e:  # wrong sheet name, corrupt file, ...
        return False, f"could not read sheet 'Monthly': {e}"
    if not {"month", "revenue"} <= set(got.columns):
        return False, f"wrong columns: {list(got.columns)}"
    got["month"] = got["month"].astype(str).str[:7]
    got = got.set_index("month")["revenue"]
    want = truth["excel"]
    if list(got.index) != list(want.index):
        return False, "months do not match the expected list"
    if (got - want).abs().max() > 0.01:
        return False, "revenue values differ from the correct totals"
    return True, "values match"


CHECKS = {"chart": check_chart, "stats": check_stats, "excel": check_excel}
OUTPUT_FILES = ["chart1.png", "summary.xlsx"]


# ---------------------------------------------------------------- model calls
def post(payload):
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def ask(model, prompt, num_ctx):
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "options": {"num_ctx": num_ctx, "temperature": 0.2},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }
    start = time.time()
    try:
        data = post(payload)
    except urllib.error.HTTPError:
        # some models reject the "think" field - retry without it
        payload.pop("think")
        data = post(payload)
    wall = time.time() - start
    text = data["message"]["content"]
    tokens = data.get("eval_count", 0)
    secs = data.get("eval_duration", 0) / 1e9
    return text, tokens, (tokens / secs if secs else 0.0), wall


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else text


def run_code(code, path):
    path.write_text(code, encoding="utf-8")
    env = dict(os.environ, MPLBACKEND="Agg")
    try:
        p = subprocess.run(
            [sys.executable, str(path.resolve())],
            cwd=WORK,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
    except subprocess.TimeoutExpired:
        return False, "", "timed out after 90 s"
    last_err = p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ""
    return p.returncode == 0, p.stdout, last_err


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3.5:4b", "gemma4:e4b", "qwen2.5:7b"])
    ap.add_argument("--num-ctx", type=int, default=4096)
    args = ap.parse_args()

    check_imports()
    WORK.mkdir(exist_ok=True)
    GEN.mkdir(exist_ok=True)
    df = make_data()
    stats_truth, monthly_truth = compute_truth(df)
    truth = {"stats": stats_truth, "excel": monthly_truth}

    results = []
    for model in args.models:
        print(f"\n=== {model} ===")
        for task, prompt in TASKS.items():
            for f in OUTPUT_FILES:
                (WORK / f).unlink(missing_ok=True)
            try:
                text, tokens, tps, wall = ask(model, prompt, args.num_ctx)
            except Exception as e:
                print(f"  {task:6} FAIL  could not get an answer: {e}")
                results.append(dict(model=model, task=task, ok=False, note=str(e), tps=0, wall=0, tokens=0))
                continue
            code = extract_code(text)
            safe = model.replace(":", "_").replace("/", "_")
            ran, stdout, err = run_code(code, GEN / f"{safe}__{task}.py")
            if ran:
                ok, note = CHECKS[task](stdout, truth)
            else:
                ok, note = False, f"code crashed: {err}"
            if ok and task == "chart":
                # keep the chart per model so you can compare them by eye
                (WORK / "chart1.png").replace(WORK / f"{safe}__chart.png")
            print(f"  {task:6} {'PASS' if ok else 'FAIL'}  {tps:4.1f} tok/s  {wall:5.0f}s  {tokens:4d} tokens  {note}")
            results.append(dict(model=model, task=task, ok=ok, note=note, tps=tps, wall=wall, tokens=tokens))

    print("\n=== SUMMARY ===")
    print(f"{'model':14} {'passed':8} {'avg tok/s':10} {'total time':10}")
    for model in args.models:
        rs = [r for r in results if r["model"] == model]
        passed = sum(r["ok"] for r in rs)
        speeds = [r["tps"] for r in rs if r["tps"]]
        avg = sum(speeds) / len(speeds) if speeds else 0
        total = sum(r["wall"] for r in rs)
        print(f"{model:14} {passed}/{len(rs):<6} {avg:<10.1f} {total:.0f}s")

    (WORK / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nGenerated code is in {GEN}  |  full results in {WORK / 'results.json'}")


if __name__ == "__main__":
    main()
