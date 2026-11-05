"""
bench_hard.py - harder, realistic analyst tasks with exact ground truth.

A messy synthetic sales file is generated (text numbers like '$1,234.50' and '10%',
day-first dates, 'N/A' values, 30 duplicate rows, a region with a single row, and a
planted effect: the West region has higher prices; Nov/Dec are strong, Feb is weak).
Each task is graded against the true answer, so a pass means the agent got it right.

Keep agent.py, analyst_tools.py and this file in one folder.

    python bench_hard.py                               # gemma4:e4b, helpers on, all tasks
    python bench_hard.py --tier 2                      # only the harder statistics tasks
    python bench_hard.py --models qwen3.5:4b gemma4:e4b
    python bench_hard.py --no-helpers                  # to see what the helpers add
Full transcripts of every run are saved in bench_hard_work/transcript_*.md
"""
import argparse
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from agent import Agent

WORK = Path("bench_hard_work")

INTRO = (
    "The file sales.csv is messy: numbers are stored as text like '$1,234.50' and '10%', "
    "dates are day-first text (dd/mm/yyyy), some values are 'N/A', and there are exact duplicate rows. "
    "Columns: Order Date, Region, Product, Units, Unit Price, Discount %, Revenue, Returned (Yes/No). "
)

TASKS = {
    "clean": INTRO + "Clean it and remove exact duplicate rows. Do NOT drop rows just because a value is missing. Print exactly one line: "
    "RESULT rows=<rows after removing duplicates> dups=<number of duplicate rows removed> revenue=<total Revenue, 2 decimals>",
    "groups": INTRO + "Exclude regions with fewer than 10 rows (after removing duplicates). Find the region with the highest "
    "mean Revenue and test whether regions differ (ANOVA or Kruskal-Wallis). Print exactly one line: "
    "RESULT top=<region> p=<p-value>",
    "regress": INTRO + "After removing duplicates and rows with missing Units, regress Revenue on Units and Unit Price "
    "(ordinary least squares with intercept). Print exactly one line: RESULT coef_units=<coefficient of Units> r2=<R-squared>",
    "outliers": INTRO + "After removing duplicates, count the Revenue outliers using the IQR rule "
    "(below Q1-1.5*IQR or above Q3+1.5*IQR). Print exactly one line: RESULT outliers=<count>",
    "trend": INTRO + "After removing duplicates, find the calendar month (format YYYY-MM) with the highest total Revenue "
    "and the one with the lowest. Print exactly one line: RESULT best=<YYYY-MM> worst=<YYYY-MM>",
}

TIER2 = {
    "chisq": INTRO + "After removing duplicates, test whether the Returned rate is associated with Product (chi-square test of "
    "independence on the Product x Returned table). Print exactly one line: RESULT chi2=<statistic> "
    "p=<p-value in scientific notation with 4 significant digits, e.g. 1.234e-08>",
    "props": INTRO + "After removing duplicates, compare the return rate (Returned == 'Yes'; each row is one order, so the rate is the share of rows) of Product E against all other products "
    "combined with a two-proportion z-test using a pooled standard error (two-sided). Print exactly one line: "
    "RESULT z=<z for E minus others> p=<p-value in scientific notation with 4 significant digits>",
    "forecast": INTRO + "After removing duplicates, compute total Revenue per calendar month (sorted by month, numbered 0,1,2,...), "
    "fit a least-squares straight line of monthly revenue against the month number, and forecast the month right after the last one. "
    "Print exactly one line: RESULT slope=<slope per month> forecast=<forecast revenue, 2 decimals>",
    "catreg": INTRO + "After removing duplicates, rows with missing Units, and the Island region, regress Revenue on Units, Unit Price and "
    "Region (dummy-coded, East as the reference level, ordinary least squares with intercept). Print exactly one line: "
    "RESULT coef_west=<coefficient of West versus East> r2=<R-squared>",
}
ALL_TASKS = {**TASKS, **TIER2}


# ------------------------------------------------------------------ data + truth
def make_data(workdir):
    rng = np.random.default_rng(42)
    n = 3000
    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 547, n), unit="D")  # to 2025-06-30
    region = rng.choice(["North", "South", "East", "West"], n, p=[0.28, 0.27, 0.25, 0.20]).astype(object)
    region[17] = "Island"  # a region with a single row
    product = rng.choice(list("ABCDE"), n)
    units = rng.integers(1, 31, n)
    units = np.where(np.isin(dates.month, [11, 12]), units + rng.integers(10, 25, n), units)
    units = np.where(dates.month == 2, np.maximum(1, units // 2), units)
    price = rng.uniform(5, 50, n)
    price = np.where(region == "West", price * 1.3, price).round(2)
    disc = rng.choice([0, 5, 10, 15], n)
    revenue = (units * price * (1 - disc / 100)).round(2)
    na = rng.random(n) < 0.02
    returned = rng.random(n) < np.where(product == "E", 0.15, 0.06)  # planted: product E is returned far more often

    base = pd.DataFrame({"date": dates, "region": region, "product": product, "units": np.where(na, np.nan, units),
                         "price": price, "revenue": revenue, "returned": returned})
    raw = pd.DataFrame(
        {
            "Order Date": dates.strftime("%d/%m/%Y"),
            "Region": region,
            "Product": product,
            "Units": np.where(na, "N/A", units.astype(str)),
            "Unit Price": [f"${p:.2f}" for p in price],
            "Discount %": [f"{d}%" for d in disc],
            "Revenue": [f"${r:,.2f}" for r in revenue],
            "Returned": np.where(returned, "Yes", "No"),
        }
    )
    raw = pd.concat([raw, raw.iloc[100:130]], ignore_index=True)  # 30 exact duplicates
    Path(workdir).mkdir(exist_ok=True)
    raw.to_csv(Path(workdir) / "sales.csv", index=False)

    means = base[base.region != "Island"].groupby("region")["revenue"].mean()
    ok = base.dropna(subset=["units"])
    A = np.column_stack([np.ones(len(ok)), ok.units, ok.price])
    beta = np.linalg.lstsq(A, ok.revenue.to_numpy(), rcond=None)[0]
    r2 = 1 - ((ok.revenue - A @ beta) ** 2).sum() / ((ok.revenue - ok.revenue.mean()) ** 2).sum()
    q1, q3 = base.revenue.quantile([0.25, 0.75])
    iqr = q3 - q1
    monthly = base.groupby(base.date.dt.strftime("%Y-%m"))["revenue"].sum()
    from scipy import stats

    chi2, p_chi = stats.chi2_contingency(pd.crosstab(base["product"], base["returned"]))[:2]
    e, o = base.loc[base["product"] == "E", "returned"], base.loc[base["product"] != "E", "returned"]
    pooled = (e.sum() + o.sum()) / (len(e) + len(o))
    z = (e.mean() - o.mean()) / np.sqrt(pooled * (1 - pooled) * (1 / len(e) + 1 / len(o)))
    slope, intercept = np.polyfit(np.arange(len(monthly)), monthly.to_numpy(), 1)
    sub = ok[ok.region != "Island"]
    D = np.column_stack([np.ones(len(sub)), sub.units, sub.price] + [(sub.region == r).astype(float) for r in ("North", "South", "West")])
    b2 = np.linalg.lstsq(D, sub.revenue.to_numpy(), rcond=None)[0]
    r2b = 1 - ((sub.revenue - D @ b2) ** 2).sum() / ((sub.revenue - sub.revenue.mean()) ** 2).sum()
    return {
        "chi2": float(chi2), "p_chi": float(p_chi), "z": float(z), "p_z": float(2 * stats.norm.sf(abs(z))),
        "slope": float(slope), "forecast": float(intercept + slope * len(monthly)), "coef_west": float(b2[5]), "r2_cat": float(r2b),
        "rows": n, "dups": 30, "revenue": round(float(base.revenue.sum()), 2),
        "top": means.idxmax(), "coef_units": float(beta[1]), "r2": float(r2),
        "outliers": int(((base.revenue < q1 - 1.5 * iqr) | (base.revenue > q3 + 1.5 * iqr)).sum()),
        "best": monthly.idxmax(), "worst": monthly.idxmin(),
    }


# ----------------------------------------------------------------------- grading
def parse_result(stdout):
    lines = re.findall(r"RESULT\s+(.+)", stdout)
    return dict(re.findall(r"(\w+)=([^\s,]+)", lines[-1])) if lines else None


def _f(x):
    return float(str(x).replace(",", "").replace("$", ""))


def grade(task, stdout, t):
    r = parse_result(stdout)
    if r is None:
        return False, "no 'RESULT ...' line printed"
    try:
        if task == "clean":
            ok = int(_f(r["rows"])) == t["rows"] and int(_f(r["dups"])) == t["dups"] and abs(_f(r["revenue"]) - t["revenue"]) < 0.02
            return ok, f"got {r} | want rows={t['rows']} dups={t['dups']} revenue={t['revenue']}"
        if task == "groups":
            p = _f(r["p"])
            return (r["top"].strip().lower() == t["top"].lower() and math.isfinite(p) and p < 0.05), f"got {r} | want top={t['top']}, p<0.05"
        if task == "regress":
            c, q = _f(r["coef_units"]), _f(r["r2"])
            return (abs(c - t["coef_units"]) < 0.01 * abs(t["coef_units"]) and abs(q - t["r2"]) < 0.005), (
                f"got {r} | want coef_units={t['coef_units']:.3f} r2={t['r2']:.4f}")
        if task == "outliers":
            return int(_f(r["outliers"])) == t["outliers"], f"got {r} | want outliers={t['outliers']}"
        if task in ("chisq", "props"):
            stat_key, p_key, tp = ("chi2", "p_chi", t["p_chi"]) if task == "chisq" else ("z", "p_z", t["p_z"])
            tv = t["chi2"] if task == "chisq" else t["z"]
            sv, pv = _f(r[stat_key]), _f(r["p"])
            return (abs(sv - tv) < 0.01 * abs(tv) and abs(pv - tp) < 0.05 * tp), f"got {r} | want {stat_key}={tv:.4f} p={tp:.4e}"
        if task == "forecast":
            sl, fc = _f(r["slope"]), _f(r["forecast"])
            return (abs(sl - t["slope"]) < 0.01 * abs(t["slope"]) and abs(fc - t["forecast"]) < 0.01 * abs(t["forecast"])), (
                f"got {r} | want slope={t['slope']:.2f} forecast={t['forecast']:.2f}")
        if task == "catreg":
            cw, q = _f(r["coef_west"]), _f(r["r2"])
            return (abs(cw - t["coef_west"]) < 0.01 * abs(t["coef_west"]) and abs(q - t["r2_cat"]) < 0.005), (
                f"got {r} | want coef_west={t['coef_west']:.3f} r2={t['r2_cat']:.4f}")
        if task == "trend":
            return (r["best"] == t["best"] and r["worst"] == t["worst"]), f"got {r} | want best={t['best']} worst={t['worst']}"
    except (KeyError, ValueError) as e:
        return False, f"could not read result ({e!r}): {r}"
    return False, "unknown task"


# -------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma4:e4b"])
    ap.add_argument("--tier", choices=["1", "2", "all"], default="all", help="1 = basics, 2 = harder statistics")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--no-helpers", action="store_true")
    args = ap.parse_args()

    truth = make_data(WORK)
    print("Planted truth:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in truth.items()})
    rows = []
    for model in args.models:
        print(f"\n=== {model} (hard tasks, helpers {'off' if args.no_helpers else 'on'}) ===")
        agent = Agent(model, workdir=WORK, num_ctx=args.num_ctx, max_steps=args.max_steps, helpers=not args.no_helpers)
        try:
            for task, prompt in {"1": TASKS, "2": TIER2, "all": ALL_TASKS}[args.tier].items():
                for stray in WORK.iterdir():  # no leftovers from earlier tasks
                    if stray.name != "sales.csv" and not stray.name.startswith(("transcript_", "session_")):
                        stray.unlink()
                agent.reset()
                t0 = time.time()
                try:
                    agent.solve(prompt)
                except Exception as e:
                    print(f"  {task:9} FAIL  agent error: {e}")
                    rows.append((model, task, False, 0, 0))
                    continue
                wall = time.time() - t0
                stdout = "\n".join(h["stdout"] for h in agent.history if not h["error"] and not h.get("warning"))
                ok, note = grade(task, stdout, truth)
                errs = sum(1 for h in agent.history if h["error"])
                print(f"  {task:9} {'PASS' if ok else 'FAIL'}  steps={agent.last_steps} errors={errs} {wall:4.0f}s  {note}")
                rows.append((model, task, ok, agent.last_steps, wall))
        finally:
            speed = agent.eval_tokens / agent.eval_secs if agent.eval_secs else 0
            agent.close()
            print(f"  average speed: {speed:.1f} tok/s")

    print("\n=== SUMMARY (hard tasks) ===")
    print(f"{'model':14} {'passed':8} {'avg steps':10} {'total time':10}")
    for model in args.models:
        rs = [r for r in rows if r[0] == model]
        if rs:
            print(f"{model:14} {sum(r[2] for r in rs)}/{len(rs):<6} {sum(r[3] for r in rs) / len(rs):<10.1f} {sum(r[4] for r in rs):.0f}s")
    print(f"\nTranscripts of every step (including errors) are in {WORK}\\transcript_*.md")


if __name__ == "__main__":
    main()
