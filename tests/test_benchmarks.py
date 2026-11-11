"""The benchmarks grade a model against planted answers. These tests check the benchmarks themselves: that correct
solutions pass, that wrong ones fail, and that the planted truth has not drifted (no model needed)."""
import os
import subprocess
import sys
import unittest

import bench_excel as X
import bench_hard as H
from _common import ROOT, InTempDir

PRE = "import pandas as pd, numpy as np\nfrom scipy import stats\nfrom analyst_tools import *\n"
LOAD = "c = clean_columns(pd.read_csv('sales.csv').drop_duplicates(), dayfirst=True)\n"


def run_solution(code, cwd):
    env = {**os.environ, "PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg"}
    p = subprocess.run([sys.executable, "-c", PRE + code], cwd=cwd, capture_output=True, text=True, env=env, timeout=180)
    return p.returncode, p.stdout, p.stderr


HARD_GOOD = {
    "clean": "df = pd.read_csv('sales.csv'); n0 = len(df); df = df.drop_duplicates(); c = clean_columns(df, dayfirst=True)\n"
             "print(f\"RESULT rows={len(c)} dups={n0 - len(c)} revenue={c['Revenue'].sum():.2f}\")",
    "groups": LOAD + "vc = c.Region.value_counts(); c = c[c.Region.isin(vc[vc >= 10].index)]\nr = compare_groups(c, 'Revenue', 'Region')\n"
              "print(f\"RESULT top={c.groupby('Region').Revenue.mean().idxmax()} p={r.p:.6f}\")",
    "regress": LOAD + "c = c.dropna(subset=['Units'])\nr = regress(c, 'Revenue', ['Units', 'Unit Price'])\nprint(f\"RESULT coef_units={r.coef.loc['Units', 'coef']:.4f} r2={r.r2:.4f}\")",
    "outliers": LOAD + "q1, q3 = c.Revenue.quantile([.25, .75]); i = q3 - q1\nprint(f\"RESULT outliers={int(((c.Revenue < q1 - 1.5 * i) | (c.Revenue > q3 + 1.5 * i)).sum())}\")",
    "trend": LOAD + "m = c.groupby(c['Order Date'].dt.strftime('%Y-%m')).Revenue.sum()\nprint(f\"RESULT best={m.idxmax()} worst={m.idxmin()}\")",
    "chisq": LOAD + "chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(c.Product, c.Returned))\nprint(f'RESULT chi2={chi2:.4f} p={p:.4e}')",
    "props": LOAD + "e = c[c.Product == 'E'].Returned == 'Yes'; o = c[c.Product != 'E'].Returned == 'Yes'; pp = (e.sum() + o.sum()) / (len(e) + len(o))\n"
             "z = (e.mean() - o.mean()) / np.sqrt(pp * (1 - pp) * (1 / len(e) + 1 / len(o)))\nprint(f'RESULT z={z:.4f} p={2 * stats.norm.sf(abs(z)):.4e}')",
    "forecast": LOAD + "m = c.groupby(c['Order Date'].dt.strftime('%Y-%m')).Revenue.sum().sort_index()\nsl, ic = np.polyfit(np.arange(len(m)), m.to_numpy(), 1)\n"
                "print(f'RESULT slope={sl:.4f} forecast={ic + sl * len(m):.2f}')",
    "catreg": LOAD + "c = c.dropna(subset=['Units']); c = c[c.Region != 'Island']\nr = regress(c, 'Revenue', ['Units', 'Unit Price', 'Region'])\n"
              "print(f\"RESULT coef_west={r.params['Region[West]']:.4f} r2={r.r2:.4f}\")",
}
HARD_WRONG = {
    "clean": "print('RESULT rows=3030 dups=0 revenue=1.00')", "groups": "print('RESULT top=North p=nan')", "regress": "print('RESULT coef_units=1.0 r2=0.1')",
    "outliers": "print('RESULT outliers=0')", "trend": "print('RESULT best=2024-01 worst=2024-02')", "chisq": "print('RESULT chi2=3.0 p=0.5')",
    "props": "print('RESULT z=0.1 p=0.9')", "forecast": "print('RESULT slope=1.0 forecast=100.00')", "catreg": "print('RESULT coef_west=1.0 r2=0.1')",
}


class HardBenchmarkTests(InTempDir):
    def setUp(self):
        super().setUp()
        self.work = self.tmp / "work"
        self.truth = H.make_data(self.work)

    def test_planted_truth_has_not_drifted(self):
        t = self.truth
        self.assertEqual((t["rows"], t["dups"], t["revenue"], t["top"], t["outliers"], t["best"], t["worst"]), (3000, 30, 1315116.2, "West", 65, "2024-12", "2025-02"))
        self.assertAlmostEqual(t["coef_units"], 26.5506, places=4)
        self.assertAlmostEqual(t["r2"], 0.8643, places=4)
        self.assertAlmostEqual(t["chi2"], 31.5004, places=4)
        self.assertAlmostEqual(t["z"], 5.5578, places=4)

    def test_correct_solutions_pass_every_task(self):
        self.assertEqual(set(HARD_GOOD), set(H.ALL_TASKS))
        for task, code in HARD_GOOD.items():
            rc, out, err = run_solution(code, self.work)
            self.assertEqual(rc, 0, f"{task}: {err[-400:]}")
            ok, note = H.grade(task, out, self.truth)
            self.assertTrue(ok, f"{task}: {note}")

    def test_wrong_solutions_fail_every_task(self):
        for task, code in HARD_WRONG.items():
            rc, out, err = run_solution(code, self.work)
            self.assertFalse(H.grade(task, out, self.truth)[0], task)

    def test_format_requirements_are_strict(self):
        self.assertFalse(H.grade("chisq", f"RESULT chi2={self.truth['chi2']:.4f} p=0.0000", self.truth)[0])  # p must be in scientific notation
        self.assertFalse(H.grade("regress", "RESULT coef_units=coef 26.55\nstd_err ... r2=0.8643", self.truth)[0])  # a table row is not a number
        self.assertFalse(H.grade("clean", "no result line here", self.truth)[0])


EXCEL_GOOD = {
    "xl_read": "df = read_excel_smart('report.xlsx', 'Summary', verbose=False)\nprint(f\"RESULT rows={len(df)} revenue={df['Revenue'].sum():.2f}\")",
    "xl_merged": "df = read_excel_smart('report.xlsx', 'Regional', verbose=False)\ng = df.groupby('Region')['Revenue'].sum()\nprint(f\"RESULT north={g['North']} west={g['West']}\")",
    "xl_write": "df = read_excel_smart('report.xlsx', 'Summary', verbose=False)\ng = df.groupby('Region', as_index=False)['Revenue'].sum()\n"
                "save_excel('summary.xlsx', {'Summary': g}, totals=True, charts=[{'kind': 'bar', 'x': 'Region', 'y': 'Revenue'}])\nprint('RESULT saved=summary.xlsx')",
    "xl_formula": "df = read_excel_smart('report.xlsx', 'Summary', verbose=False)\nsave_excel('margin.xlsx', {'Data': df}, formulas={'Avg Price': '{Revenue}/{Units}'})\nprint('RESULT saved=margin.xlsx')",
}
EXCEL_WRONG = {
    "xl_read": "import pandas as pd\ndf = pd.read_excel('report.xlsx', 'Summary')\nprint(f\"RESULT rows={len(df)} revenue=1.00\")",  # naive read: title rows break it
    "xl_merged": "print('RESULT north=100 west=300')",  # merged cells not filled down
    "xl_write": "import pandas as pd\npd.DataFrame({'Region': ['North', 'South', 'East', 'West'], 'Revenue': [1, 2, 3, 4]}).to_excel('summary.xlsx', sheet_name='Summary', index=False)\nprint('RESULT saved=summary.xlsx')",
    "xl_formula": "import pandas as pd\ndf = pd.read_excel('report.xlsx', 'Summary', header=3).iloc[:40]\ndf['Avg Price'] = df['Revenue'] / df['Units']\n"
                  "df.to_excel('margin.xlsx', sheet_name='Data', index=False)\nprint('RESULT saved=margin.xlsx')",  # pasted values, not formulas
}


class ExcelBenchmarkTests(InTempDir):
    def setUp(self):
        super().setUp()
        self.work = self.tmp / "work"
        self.truth = X.make_data(self.work)

    def clear_outputs(self):
        for f in os.listdir(self.work):
            if f != "report.xlsx":
                os.remove(self.work / f)

    def test_planted_truth(self):
        self.assertEqual((self.truth["rows"], self.truth["revenue"], self.truth["north"], self.truth["west"]), (40, 11412.26, 450.0, 950.0))
        self.assertAlmostEqual(self.truth["per_region"]["West"], 2513.01, places=2)

    def test_correct_solutions_pass_and_wrong_ones_fail(self):
        for label, solutions, want in (("good", EXCEL_GOOD, True), ("wrong", EXCEL_WRONG, False)):
            for task, code in solutions.items():
                self.clear_outputs()
                rc, out, err = run_solution(code, self.work)
                ok, note = X.grade(task, out, self.work, self.truth) if rc == 0 else (False, err[-300:])
                self.assertEqual(ok, want, f"{label} {task}: {note}")

    def test_file_graders_catch_specific_mistakes(self):
        self.clear_outputs()
        run_solution("save_excel('summary.xlsx', {'Data': pd.DataFrame({'Region': ['North'], 'Revenue': [1.0]})}, totals=True)", self.work)
        self.assertIn("no sheet named 'Summary'", X.grade("xl_write", "", self.work, self.truth)[1])
        self.clear_outputs()
        run_solution("df = read_excel_smart('report.xlsx', 'Summary', verbose=False)\n"
                     "g = df.groupby('Region', as_index=False)['Revenue'].sum().rename(columns={'Revenue': 'Total Revenue'})\n"
                     "save_excel('summary.xlsx', {'Summary': g}, totals=True)", self.work)
        self.assertIn("header should contain Region and Revenue", X.grade("xl_write", "", self.work, self.truth)[1])


if __name__ == "__main__":
    unittest.main()
