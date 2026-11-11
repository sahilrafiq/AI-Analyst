"""Tests for agent.py: the guards and the loop, driven by scripted model replies and a fake kernel (no Ollama needed)."""
import os
import re
import unittest

import pandas as pd

import agent
import bench_excel as X
import bench_hard as H
from _common import InTempDir, WithFakeAgent, captured, resp


def assistant_turns(payload):
    return sum(m["role"] == "assistant" for m in payload["messages"])


class WarningAndFormattingTests(unittest.TestCase):
    def test_failed_results_are_recognised(self):
        for bad in ("RESULT p=nan", "RESULT coef_west=None r2=0.86", "RESULT z=inf p=0", "Test: Welch ANOVA: F=nan, p=nan -> x"):
            self.assertIn("nan/inf/None", agent.result_warning(bad) or "", bad)
        for bad in ("RESULT z=15 p=0.0000e+00", "p=0.0", "p-value: 0", "p = 0.00"):
            self.assertIn("exactly 0", agent.result_warning(bad) or "", bad)

    def test_real_results_are_left_alone(self):
        for ok in ("RESULT z=5.5578 p=2.7317e-08", "p=0.0412", "p=0.0001", "Test: Welch ANOVA: F=1.2, p<0.0001", "| Intercept | -429.2 | 8.176 | 0 |",
                   "   price  units\n0   NaN   3\n1   4.0   5", "RESULT coef_west=-4.8455 r2=0.8645"):
            self.assertIsNone(agent.result_warning(ok), ok)

    def test_long_output_keeps_result_blocks_whole(self):
        from analyst_tools import compare_groups, regress
        import numpy as np

        rng = np.random.default_rng(5)
        df = pd.DataFrame({"g": rng.choice(["Australia", "Canada", "France"], 600), "v": rng.lognormal(6, 1, 600), "x": rng.normal(size=600)})
        df["y"] = 2 * df.x + rng.normal(size=600)
        _, out = captured(lambda: (compare_groups(df, "v", "g"), regress(df, "y", ["x"])))
        filler = "\n".join(f"- col{i} [int64] missing 0.0%, unique {i}, min 1, median 9, mean 267.3" for i in range(150))
        condensed = agent.condense(filler + "\n" + out + "\n" + filler, 1800)
        for must_keep in ("Test: Welch ANOVA", "Effect size", "Australia: n=", "France: n=", "OLS: y ~ x", "R2=", "Intercept", "Diagnostics:"):
            self.assertIn(must_keep, condensed)
        self.assertLess(len(condensed), 4500)

    def test_tracebacks_are_shortened_to_the_models_own_line(self):
        tb = ("---------------------------------------------------------------------------\n"
              "TypeError                                 Traceback (most recent call last)\n"
              "Cell In[6], line 12\n     10 y = monthly_revenue['Revenue']\n---> 12 model = regress(monthly_revenue, y=y, X=['Month_Num'])\n"
              "     13 slope = model.params['Month_Num']\n\n"
              "File D:\\ai-analyst\\analyst_tools.py:482, in regress(df, y, X, robust, alpha)\n    480 d = df[[y] + X].dropna().copy()\n"
              "File ~\\pandas\\core\\frame.py:4108, in DataFrame.__getitem__(self, key)\n-> 351 'Could not get source, probably due dynamically evaluated source code.'\n\n"
              "TypeError: unhashable type: 'Series'")
        out = agent.shorten_traceback(tb)
        self.assertIn("---> 12 model = regress", out)
        self.assertTrue(out.endswith("TypeError: unhashable type: 'Series'"))
        self.assertNotIn("Could not get source", out)
        self.assertTrue(agent.shorten_traceback("Traceback (most recent call last)\nFile x.py:1\n  a\nValueError: boom").endswith("ValueError: boom"))
        own = "Cell In[3], line 4\n      3 x = 1\n----> 4 compare_proportions(df, 'a', 'b', 'Z')\n\nValueError: 'Z' is not a value of 'Product'. Available: ['A', 'B']"
        self.assertIn("Available: ['A', 'B']", agent.shorten_traceback(own))

    def test_pattern_detectors(self):
        for bad in ("df = df.iloc[:-1]", "x = df.head(-1)", "d2 = df[:-1]", "df.drop(df.index[-1], inplace=True)", "df=df.iloc[: -2]"):
            self.assertTrue(agent.ROW_DROP.search(bad), bad)
        for ok in ("df.iloc[:5]", "df.head(10)", "df.iloc[-1]", "df.iloc[1:]"):
            self.assertFalse(agent.ROW_DROP.search(ok), ok)
        self.assertEqual(agent.SHEET_SPEC.findall("one sheet named 'Summary'"), ["Summary"])
        self.assertEqual(agent.COLUMN_SPEC.findall("a new last column named 'Avg Price' whose"), ["Avg Price"])

    def test_system_prompt_is_complete(self):
        self.assertIn("Also installed:", agent.package_line())
        self.assertIn("scipy", agent.package_line())


class AgentLoopTests(WithFakeAgent):
    def setUp(self):
        super().setUp()
        self.ws = self.tmp / "ws"
        self.ws.mkdir()
        self.truth_excel = X.make_data(self.ws)  # report.xlsx
        self.truth_hard = H.make_data(self.ws)  # sales.csv (3,030 rows incl. 30 duplicates)

    def agent(self, **kw):
        return self.new_agent(self.ws, **kw)

    def test_system_prompt_lists_all_tool_families(self):
        prompt = self.agent().messages[0]["content"]
        self.assertNotIn("@@", prompt)
        for name in ("compare_groups", "read_excel_smart", "save_excel", "trend_forecast", "build_powerbi_package", "crosstab_test"):
            self.assertIn(name, prompt)

    def test_same_error_twice_then_three_times_changes_the_advice(self):
        seen = []

        def post(payload):
            n = assistant_turns(payload)
            seen.append(payload["messages"][-1]["content"])
            if n < 3:
                return resp("```python\nraise TypeError('boom')\n```")
            self.assertIn("ABANDON this approach", payload["messages"][-1]["content"]) if n == 3 else None
            return resp("```python\nprint('RESULT ok=1')\n```") if n == 3 else resp("Done.")

        self.script(post)
        ag = self.agent(max_steps=8)
        ag.solve("t")
        self.assertIn("same error again", seen[2])
        self.assertNotIn("ABANDON", seen[2])
        self.assertIn("ABANDON", seen[3])
        self.assertEqual(sum(1 for h in ag.history if h["error"]), 3)

    def test_nan_result_is_rejected_and_kept_out_of_the_report(self):
        report_prompts = []

        def post(payload):
            if payload["messages"][0]["content"].startswith("You write clear"):
                report_prompts.append(payload["messages"][1]["content"])
                return resp("## Summary\n- Revenue mean 754.4.\n")
            n = assistant_turns(payload)
            if n == 0:
                return resp("```python\nprint('RESULT p=nan')\n```")
            if n == 1:
                self.assertIn("contains nan/inf/None", payload["messages"][-1]["content"])
                return resp("```python\nprint('Revenue mean 754.4')\n```")
            return resp("Done.")

        self.script(post)
        ag = self.agent()
        ag.solve("t")
        self.assertTrue(ag.history[0].get("warning"))
        ag.write_report("T")
        self.assertNotIn("p=nan", report_prompts[0])

    def test_p_value_printed_as_zero_is_rejected(self):
        def post(payload):
            n = assistant_turns(payload)
            if n == 0:
                return resp("```python\nfrom scipy.stats import norm\nz=15.17\nprint(f'RESULT z={z} p={2*(1-norm.cdf(z)):.4e}')\n```")
            if n == 1:
                self.assertIn("exactly 0", payload["messages"][-1]["content"])
                self.assertIn(".sf", payload["messages"][-1]["content"])
                return resp("```python\nfrom scipy.stats import norm\nprint(f'RESULT z=15.17 p={2*norm.sf(15.17):.4e}')\n```")
            return resp("Done.")

        self.script(post)
        ag = self.agent()
        ag.solve("t")
        self.assertTrue(ag.history[0].get("warning") and not ag.history[1].get("warning"))

    CLEAN = "```python\ndf = clean_columns(pd.read_csv('sales.csv'), dayfirst=True)\nprint(f'RESULT rows={len(df)}')\n```"

    def test_identical_successful_code_is_not_rerun_and_made_up_numbers_are_flagged(self):
        self.script(lambda p: resp(self.CLEAN) if assistant_turns(p) < 2 else resp("Done: 2970 rows."))
        ag = self.agent()
        out = ag.solve("count rows")
        self.assertEqual(len(ag.history), 1)  # the second identical block was not executed
        flagged = out.split("[CHECK:")[1]
        self.assertIn("2970", flagged)  # no output ever produced this number
        self.assertNotIn("3030", flagged)  # a number that did appear in the output is fine

    def test_a_model_that_keeps_resending_the_same_code_is_stopped(self):
        self.script(lambda p: resp(self.CLEAN))
        self.assertTrue(self.agent().solve("count rows").startswith("Stopped: the model kept resending"))

    def test_required_result_line_must_be_printed_by_code(self):
        def post(payload):
            n = assistant_turns(payload)
            if n == 0:
                return resp("```python\nx = 1\n```")
            if n == 1:
                return resp("RESULT a=1")  # only in the answer, never printed
            if n == 2:
                self.assertIn("never printed it", payload["messages"][-1]["content"])
                return resp("```python\nprint('RESULT a=1')\n```")
            return resp("Done.")

        self.script(post)
        ag = self.agent()
        ag.solve("Do x. Print exactly one line: RESULT a=<a>")
        self.assertIn("RESULT a=1", ag.history[-1]["stdout"])
        self.script(lambda p: resp("RESULT x=1"))  # a stubborn model is asked only once
        self.assertTrue(self.agent().solve("Do x. Print exactly one line: RESULT a=<a>").startswith("RESULT x=1"))

    def test_missing_package_gets_a_specific_hint(self):
        def post(payload):
            n = assistant_turns(payload)
            if n == 0:
                return resp("```python\nimport a_package_that_does_not_exist\n```")
            if n == 1:
                self.assertIn("NOT installed", payload["messages"][-1]["content"])
                return resp("```python\nprint('RESULT ok=1')\n```")
            return resp("Done.")

        self.script(post)
        self.assertEqual(sum(1 for h in self.agent().history if h["error"]), 0)
        ag = self.agent()
        ag.solve("t")
        self.assertEqual(sum(1 for h in ag.history if h["error"]), 1)

    def test_cutting_rows_off_a_cleaned_excel_table_is_blocked(self):
        def post(payload):
            n = assistant_turns(payload)
            if n == 0:
                return resp("```python\ndf = read_excel_smart('report.xlsx', sheet_name='Summary')\ndf = df.iloc[:-1]\nprint(len(df))\n```")
            if n == 1:
                self.assertIn("this code was NOT run", payload["messages"][-1]["content"])
                return resp("```python\ndf = read_excel_smart('report.xlsx', sheet_name='Summary')\nprint(len(df))\n```")
            return resp("40 rows.")

        self.script(post)
        ag = self.agent()
        ag.solve("count rows")
        self.assertEqual(len(ag.history), 1)
        self.assertTrue(ag.history[0]["stdout"].strip().endswith("40"))
        # ordinary slicing of data that no helper cleaned is allowed
        self.script(lambda p: resp("```python\nprint(len(pd.DataFrame({'a': [1, 2, 3]}).iloc[:-1]))\n```") if assistant_turns(p) == 0 else resp("Two."))
        other = self.agent()
        other.solve("slice")
        self.assertEqual(other.history[0]["stdout"].strip(), "2")

    BASE = "df = read_excel_smart('report.xlsx', sheet_name='Summary')\ng = df.groupby('Region', as_index=False)['Revenue'].sum()\n"
    SAVE = "save_excel('summary.xlsx', g, totals=True, charts=[{'kind':'bar','x':'Region','y':'%s'}]%s)\nprint('RESULT saved=summary.xlsx')"

    def test_requested_sheet_name_is_enforced_on_created_workbooks(self):
        def post(payload):
            n = assistant_turns(payload)
            if n == 0:
                return resp("```python\n" + self.BASE + self.SAVE % ("Revenue", "") + "\n```")  # default sheet name 'Data'
            if n == 1:
                return resp("Saved.")
            if n == 2:
                self.assertIn("no sheet named 'Summary'", payload["messages"][-1]["content"])
                return resp("```python\n" + self.BASE + self.SAVE % ("Revenue", ", sheet_name='Summary'") + "\n```")
            return resp("Done.")

        self.script(post)
        self.agent(max_steps=6).solve(X.TASKS["xl_write"])
        self.assertTrue(X.grade("xl_write", "", self.ws, self.truth_excel)[0])

    def test_columns_named_in_the_task_must_not_be_renamed(self):
        def post(payload):
            n = assistant_turns(payload)
            if n == 0:
                return resp("```python\n" + self.BASE + "g = g.rename(columns={'Revenue': 'Total Revenue'})\n" + self.SAVE % ("Total Revenue", ", sheet_name='Summary'") + "\n```")
            if n == 1:
                return resp("Saved summary.xlsx.")
            if n == 2:
                self.assertIn("no column named 'Revenue'", payload["messages"][-1]["content"])
                self.assertIn("do not rename", payload["messages"][-1]["content"])
                return resp("```python\n" + self.BASE + self.SAVE % ("Revenue", ", sheet_name='Summary'") + "\n```")
            return resp("Done.")

        self.script(post)
        self.agent(max_steps=6).solve(X.TASKS["xl_write"])
        self.assertTrue(X.grade("xl_write", "", self.ws, self.truth_excel)[0])

    def test_name_check_ignores_input_files_and_tasks_without_naming(self):
        self.script(lambda p: resp("Fine."))
        self.assertEqual(self.agent().solve(X.TASKS["xl_read"]), "Fine.")
        names = lambda t: [x for m in agent.COLUMNS_LIST_SPEC.finditer(t) for x in re.split(r"\s*(?:,|\band\b)\s*", m.group(1)) if x and x.lower() not in agent._NAME_STOP]
        self.assertEqual(names(X.TASKS["xl_write"]), ["Region", "Revenue"])
        self.assertEqual((names(X.TASKS["xl_formula"]), names(X.TASKS["xl_read"])), ([], []))
        self.assertEqual(names("columns named Region and a Total row"), ["Region"])

    def test_report_carries_its_own_verification_notes(self):
        from docx import Document

        md = "# Ignored Title\n## Summary\n- Welch ANOVA F=137.54, p<0.0001.\n- Sales volume reached 987,654 units for `sales\\_summary.xlsx`.\nCustomer age averages 267.3 years.\n"

        def post(payload):
            if payload["messages"][0]["content"].startswith("You write clear"):
                return resp(md)
            if assistant_turns(payload) == 0:
                return resp("```python\nprint('Welch ANOVA F=137.54 p<0.0001')\nprint('Unit_Cost mean 267.3')\n```")
            return resp("Done.")

        self.script(post)
        ag = self.agent()
        ag.solve("analyse")
        text = ag.write_report("Sales by country")
        self.assertIn("CHECK BEFORE SHARING", text)
        stem = sorted(f for f in os.listdir(self.ws) if f.startswith("report_") and f.endswith(".md"))[-1]
        notes = (self.ws / stem).read_text(encoding="utf-8").split("## Verification notes")[1]
        self.assertIn("987,654", notes)
        self.assertIn("Unit_Cost", notes)  # the mislabelled 267.3
        self.assertNotIn("137.54", notes)  # real numbers are not flagged
        doc = Document(str(self.ws / stem.replace(".md", ".docx")))
        self.assertEqual([p.text for p in doc.paragraphs if p.style.name in ("Title", "Heading 1")], ["Sales by country"])
        self.assertIn("sales_summary.xlsx", "\n".join(p.text for p in doc.paragraphs))

    def test_workspace_description_for_csv_and_excel(self):
        d = agent.describe_workspace(self.ws)
        self.assertIn("table header on row 4, then 40 data rows", d)
        self.assertIn("NOT part of the 40 data rows", d)
        self.assertIn("sales.csv (csv)", d)
        self.assertIn("Order Date:", d)

    def test_agent_builds_a_power_bi_package(self):
        code = ("```python\ndf = clean_columns(pd.read_csv('sales.csv'), dayfirst=True).drop_duplicates()\n"
                "pkg = build_powerbi_package(df, 'powerbi_out', fact='Sales', date='Order Date', value='Revenue',\n"
                "                            dims={'Product': ['Product'], 'Region': ['Region']}, rank_by=('Product', 'Product'))\n"
                "print('RESULT files=' + str(len(pkg.files)))\n```")
        self.script(lambda p: resp(code) if assistant_turns(p) == 0 else resp("Package written."))
        ag = self.agent()
        ag.solve("Prepare sales.csv for Power BI")
        self.assertFalse(ag.history[0]["error"], ag.history[0]["error"])
        out = self.ws / "powerbi_out"
        self.assertTrue({"Sales.csv", "Product.csv", "Region.csv", "Date.csv", "load_queries.m", "measures.dax", "expected_values.xlsx", "README_PowerBI.md"} <= set(os.listdir(out)))
        sales = pd.read_csv(out / "Sales.csv")
        self.assertEqual(len(sales), 3000)
        self.assertAlmostEqual(sales.Revenue.sum(), 1315116.20, places=2)


if __name__ == "__main__":
    unittest.main()
