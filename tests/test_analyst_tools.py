"""Tests for analyst_tools.py: cleaning, statistics helpers (checked against scipy/numpy), charts and Excel."""
import shutil
import subprocess
import unittest

import numpy as np
import pandas as pd
from scipy import stats

from _common import InTempDir, captured
from analyst_tools import (Result, clean_columns, compare_groups, compare_proportions, crosstab_test, inspect_workbook, profile,
                           quick_chart, read_excel_smart, regress, save_excel, top_correlations, trend_forecast)
from analyst_tools import _welch_anova, _holm


class CleanColumnsTests(InTempDir):
    def messy(self):
        rng = np.random.default_rng(7)
        n = 300
        raw = pd.DataFrame({
            "Order Date": rng.choice(["25/12/2025", "03/01/2026", "15-02-2026", "07/03/2026"], n),
            "Revenue ($)": [f"${v:,.2f}" for v in rng.uniform(100, 5000, n)],
            "Margin %": [f"{v:.1f}%" for v in rng.uniform(5, 40, n)],
            "Region ": rng.choice(["North", "South", "East", " West "], n),
            "Units": rng.integers(1, 50, n),
            "Notes": rng.choice(["ok", "n/a", "", "late"], n),
        })
        raw.loc[5, "Revenue ($)"] = "oops"
        return pd.concat([raw, raw.iloc[:3]], ignore_index=True)

    def test_names_kept_by_default_and_renamed_on_request(self):
        raw = self.messy()
        kept, _ = captured(clean_columns, raw, dayfirst=True)
        self.assertEqual(list(kept.columns), list(raw.columns))
        renamed, log = captured(clean_columns, raw, dayfirst=True, rename=True)
        self.assertEqual(list(renamed.columns), ["order_date", "revenue", "margin", "region", "units", "notes"])
        self.assertIn("renamed columns", log)

    def test_text_numbers_dates_and_missing_tokens(self):
        c, log = captured(clean_columns, self.messy(), dayfirst=True, rename=True)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(c.order_date))
        self.assertTrue(c.order_date.dt.day.isin([25, 3, 15, 7]).all() and c.order_date.dt.month.isin([12, 1, 2, 3]).all())
        self.assertEqual(c.revenue.dtype, float)
        self.assertEqual(int(c.revenue.isna().sum()), 1)  # the 'oops' value
        self.assertLess(c.margin.max(), 0.41)  # "40%" -> 0.40
        self.assertEqual(set(c.region.unique()), {"North", "South", "East", "West"})  # whitespace stripped
        self.assertGreater(int(c.notes.isna().sum()), 0)  # 'n/a' and '' become missing
        self.assertIn("duplicate rows", log)
        self.assertEqual(len(c), 303)  # duplicates are reported, never removed

    def run_clean(self, raw, **kw):
        return captured(clean_columns, raw, **kw)

    def test_day_first_detected(self):
        dates = pd.to_datetime(["2013-11-26", "2014-02-03", "2015-12-25", "2012-07-04", "2016-01-13", "2011-03-02"])
        out, log = self.run_clean(pd.DataFrame({"Date": dates.strftime("%d/%m/%Y")}))
        self.assertEqual(list(out.Date), list(dates))
        self.assertIn("day-first, detected", log)
        self.assertNotIn("WARNING", log)

    def test_month_first_detected(self):
        dates = pd.to_datetime(["2013-11-26", "2014-02-03", "2015-12-25", "2012-07-04"])
        out, log = self.run_clean(pd.DataFrame({"Date": dates.strftime("%m/%d/%Y")}))
        self.assertEqual(list(out.Date), list(dates))
        self.assertIn("month-first, detected", log)

    def test_ambiguous_dates_warn_and_explicit_choice_is_silent(self):
        amb = pd.DataFrame({"Date": ["03/04/2013", "05/06/2014", "07/08/2015"]})
        out, log = self.run_clean(amb)
        self.assertEqual(out.Date[0], pd.Timestamp("2013-03-04"))
        self.assertIn("WARNING", log)
        out, log = self.run_clean(amb, dayfirst=True)
        self.assertEqual(out.Date[0], pd.Timestamp("2013-04-03"))
        self.assertNotIn("WARNING", log)

    def test_explicit_choice_contradicted_by_data_warns(self):
        month_first = pd.DataFrame({"Date": ["11/26/2013", "02/03/2014"]})
        self.assertIn("look month-first", self.run_clean(month_first, dayfirst=True)[1])
        day_first = pd.DataFrame({"Date": ["26/11/2013", "03/02/2014"]})
        self.assertIn("look day-first", self.run_clean(day_first, dayfirst=False)[1])

    def test_mixed_formats_warn_and_iso_is_untouched(self):
        mixed = pd.DataFrame({"Date": ["25/12/2013", "12/25/2014", "03/04/2015", "26/01/2016", "01/27/2017"]})
        self.assertIn("mixed", self.run_clean(mixed)[1])
        out, log = self.run_clean(pd.DataFrame({"Date": ["2024-12-31", "2024-01-02"]}))
        self.assertEqual(out.Date[0], pd.Timestamp("2024-12-31"))
        self.assertNotIn("WARNING", log)


class ProfileTests(InTempDir):
    def test_flags_numbers_stored_as_text_and_infinite_values(self):
        df = pd.DataFrame({"a": ["1,200", "3,400", "5,600"] * 4, "b": [1.0, np.inf] * 6})
        _, out = captured(profile, df)
        self.assertIn("stored as text", out)
        self.assertIn("infinite", out)


class CompareGroupsTests(InTempDir):
    def test_two_groups_welch_matches_scipy(self):
        rng = np.random.default_rng(3)
        df = pd.DataFrame({"g": rng.choice(list("ABC"), 240), "v": rng.normal(10, 3, 240)})
        r, out = captured(compare_groups, df, "v", "g", groups=["A", "B"])
        a, b = df.v[df.g == "A"].to_numpy(), df.v[df.g == "B"].to_numpy()
        t = stats.ttest_ind(a, b, equal_var=False)
        self.assertEqual(r.test, "welch")
        self.assertAlmostEqual(r.t, t.statistic, places=9)
        self.assertAlmostEqual(r.p, t.pvalue, places=9)
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        self.assertAlmostEqual(r.d, (a.mean() - b.mean()) / sp, places=9)  # Cohen's d with pooled SD
        self.assertIn("Cohen's d", out)

    def test_three_groups_welch_anova_with_posthoc(self):
        rng = np.random.default_rng(5)
        df = pd.DataFrame({"g": rng.choice(list("ABC"), 240), "v": rng.normal(10, 3, 240)})
        df.loc[df.g == "C", "v"] += 2.5
        r, _ = captured(compare_groups, df, "v", "g")
        vals = [df.v[df.g == k] for k in "ABC"]
        ssb = sum(len(v) * (v.mean() - df.v.mean()) ** 2 for v in vals)
        self.assertEqual(r.test, "anova")
        self.assertTrue(r.significant)
        self.assertIn("posthoc", r)
        self.assertAlmostEqual(r.effect_size, ssb / ((df.v - df.v.mean()) ** 2).sum(), places=9)  # eta squared

    def test_welch_anova_with_two_groups_equals_welch_t(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(0, 1, 40), rng.normal(0.5, 2, 55)
        f, d1, d2, p = _welch_anova([a, b])
        t = stats.ttest_ind(a, b, equal_var=False)
        self.assertAlmostEqual(f, t.statistic ** 2, places=9)
        self.assertAlmostEqual(p, t.pvalue, places=9)

    def test_holm_adjustment(self):
        np.testing.assert_allclose(_holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_skewed_small_samples_switch_to_nonparametric_and_test_can_be_forced(self):
        rng = np.random.default_rng(11)
        sk = pd.DataFrame({"g": ["x"] * 20 + ["y"] * 20, "v": list(rng.lognormal(0, 1.8, 20)) + list(rng.lognormal(1.0, 1.8, 20))})
        for k in "xy":
            self.assertLess(stats.shapiro(sk.v[sk.g == k]).pvalue, 0.05)  # test data really is non-normal
        self.assertEqual(captured(compare_groups, sk, "v", "g")[0].test, "mannwhitney")
        self.assertEqual(captured(compare_groups, sk, "v", "g", test="welch")[0].test, "welch")
        sk3 = pd.DataFrame({"g": list("abc") * 8, "v": rng.lognormal(0, 1.8, 24)})
        self.assertEqual(captured(compare_groups, sk3, "v", "g")[0].test, "kruskal")

    def test_infinite_values_are_ignored_with_a_warning_never_nan(self):
        rng = np.random.default_rng(3)
        df = pd.DataFrame({"g": rng.choice(list("ABCD"), 4000), "v": rng.lognormal(6, 1, 4000)})
        df.loc[5, "v"] = np.inf
        r, out = captured(compare_groups, df, "v", "g")
        self.assertTrue(np.isfinite(r.p) and np.isfinite(r.statistic))
        self.assertIn("infinite", out)

    def test_clear_errors_for_bad_arguments(self):
        df = pd.DataFrame({"g": list("ABC") * 30, "v": np.arange(90.0)})
        with self.assertRaisesRegex(ValueError, "group NAMES"):
            compare_groups(df, "v", "g", groups=[df[df.g == "A"].v, df[df.g == "B"].v])
        with self.assertRaisesRegex(ValueError, "not found"):
            compare_groups(df, "v", "g", groups=["A", "Z"])
        with self.assertRaisesRegex(KeyError, "not found"):
            compare_groups(df, "nope", "g")


class RegressTests(InTempDir):
    def frame(self):
        rng = np.random.default_rng(1)
        reg = pd.DataFrame({"x1": rng.normal(size=200), "x2": rng.normal(size=200), "cat": rng.choice(["a", "b", "c"], 200)})
        reg["y"] = 2 * reg.x1 - reg.x2 + (reg.cat == "b") * 1.5 + rng.normal(size=200) * 0.5
        return reg

    def test_coefficients_match_least_squares_and_categories_are_named(self):
        reg = self.frame()
        r, out = captured(regress, reg, "y", ["x1", "x2", "cat"])
        design = np.column_stack([np.ones(200), reg.x1, reg.x2, (reg.cat == "b").astype(float), (reg.cat == "c").astype(float)])
        beta = np.linalg.lstsq(design, reg.y.to_numpy(), rcond=None)[0]
        np.testing.assert_allclose(r.coef["coef"].to_numpy(), beta)
        self.assertEqual(list(r.coef.index), ["Intercept", "x1", "x2", "cat[b]", "cat[c]"])
        self.assertIn("Reference levels: cat=a", out)
        self.assertAlmostEqual(r.params["b"], r.params["cat[b]"])  # short name resolves
        self.assertAlmostEqual(r.params.get("c"), r.params["cat[c]"])
        self.assertEqual(r.params.get("nope", -1), -1)
        with self.assertRaisesRegex(KeyError, "Available"):
            r.params["nope"]

    def test_single_predictor_matches_linregress(self):
        reg = self.frame()
        r, _ = captured(regress, reg, "y", "x1")
        lr = stats.linregress(reg.x1, reg.y)
        self.assertAlmostEqual(r.coef.loc["x1", "coef"], lr.slope, places=9)
        self.assertAlmostEqual(r.coef.loc["x1", "std_err"], lr.stderr, places=9)
        self.assertAlmostEqual(r.coef.loc["x1", "p"], lr.pvalue, places=9)
        self.assertAlmostEqual(r.r2, lr.rvalue ** 2, places=9)
        self.assertAlmostEqual(r.f_p, lr.pvalue, places=9)

    def test_robust_standard_errors_match_brute_force_hc3(self):
        reg = self.frame()
        X = np.column_stack([np.ones(200), reg.x1])
        y = reg.y.to_numpy()
        inv = np.linalg.inv(X.T @ X)
        H = X @ inv @ X.T
        e = y - H @ y
        meat = sum((e[i] / (1 - H[i, i])) ** 2 * np.outer(X[i], X[i]) for i in range(200))
        cov = inv @ meat @ inv
        r, _ = captured(regress, reg, "y", "x1", robust=True)
        self.assertAlmostEqual(r.coef.loc["x1", "std_err"], np.sqrt(cov[1, 1]), places=9)

    def test_diagnostics(self):
        rng = np.random.default_rng(2)
        het = pd.DataFrame({"x": rng.uniform(1, 10, 400)})
        het["y"] = het.x + rng.normal(size=400) * het.x
        hom = pd.DataFrame({"x": rng.uniform(1, 10, 400)})
        hom["y"] = hom.x + rng.normal(size=400)
        self.assertLess(captured(regress, het, "y", "x")[0].bp_p, 0.01)
        self.assertGreater(captured(regress, hom, "y", "x")[0].bp_p, 0.01)
        cl = pd.DataFrame({"a": rng.normal(size=300)})
        cl["b"] = cl.a + rng.normal(size=300) * 0.1
        cl["y"] = cl.a + rng.normal(size=300)
        r, _ = captured(regress, cl, "y", ["a", "b"])
        self.assertAlmostEqual(r.max_vif, 1 / (1 - np.corrcoef(cl.a, cl.b)[0, 1] ** 2), places=6)
        cl["c"] = cl.a * 2  # perfectly collinear: reported, not a crash
        self.assertIn("collinear", captured(regress, cl, "y", ["a", "c"])[1])

    def test_true_false_dummies_are_numbers_and_predict_reproduces_fitted(self):
        reg = self.frame()
        dummies = pd.get_dummies(reg, columns=["cat"], prefix="cat")
        r, _ = captured(regress, dummies, "y", ["x1", "cat_b", "cat_c"])
        self.assertIn("cat_b", r.params.index)  # a bool column keeps its name, no 'cat_b[True]'
        full, _ = captured(regress, reg, "y", ["x1", "x2", "cat"])
        np.testing.assert_allclose(full.predict(reg.head(50)).to_numpy(), full.fitted[:50])

    def test_argument_errors_are_explained(self):
        reg = self.frame()
        self.assertIsInstance(captured(regress, reg, y=reg["y"], X=["x1"])[0], Result)  # a Series from the frame is accepted
        with self.assertRaisesRegex(TypeError, "column NAME"):
            regress(reg, y=[1, 2], X=["x1"])
        with self.assertRaisesRegex(KeyError, "Available columns"):
            regress(reg, y="Y", X=["x1"])
        with self.assertRaisesRegex(ValueError, "infinite"):
            regress(reg.assign(x1=reg.x1.where(reg.index != 3, np.inf)), "y", ["x1"])
        with self.assertRaisesRegex(AttributeError, "Available"):
            captured(regress, reg, "y", "x1")[0].predicted


class CategoricalAndTrendTests(InTempDir):
    def test_crosstab_test_matches_scipy(self):
        rng = np.random.default_rng(4)
        df = pd.DataFrame({"a": rng.choice(list("xyz"), 600), "b": rng.choice(["p", "q"], 600)})
        r, _ = captured(crosstab_test, df, "a", "b")
        chi2, p, dof, _ = stats.chi2_contingency(pd.crosstab(df.a, df.b))
        self.assertAlmostEqual(r.chi2, chi2, places=9)
        self.assertAlmostEqual(r.p, p, places=9)
        self.assertEqual(r.dof, dof)
        self.assertAlmostEqual(r.cramers_v, np.sqrt(chi2 / (600 * 1)), places=9)

    def test_compare_proportions_pooled_z_and_aliases(self):
        rng = np.random.default_rng(6)
        df = pd.DataFrame({"p": rng.choice(list("ABCDE"), 3000), "r": rng.random(3000)})
        df["flag"] = np.where(df.r < np.where(df.p == "E", 0.15, 0.06), "Yes", "No")
        r, _ = captured(compare_proportions, df, "flag", "p", "E")
        e, o = df.flag[df.p == "E"] == "Yes", df.flag[df.p != "E"] == "Yes"
        pooled = (e.sum() + o.sum()) / (len(e) + len(o))
        z = (e.mean() - o.mean()) / np.sqrt(pooled * (1 - pooled) * (1 / len(e) + 1 / len(o)))
        self.assertAlmostEqual(r.z, z, places=9)
        self.assertAlmostEqual(r.p, 2 * stats.norm.sf(abs(z)), places=12)
        for alias in ("Other", "others", "All others"):
            self.assertAlmostEqual(captured(compare_proportions, df, "flag", "p", "E", b=alias)[0].z, z, places=12)
        with self.assertRaisesRegex(ValueError, "not a value"):
            compare_proportions(df, "flag", "p", "Z")
        with self.assertRaisesRegex(ValueError, "leave b out"):
            compare_proportions(df, "flag", "p", "E", b="Zzz")

    def test_proportion_p_value_does_not_underflow_and_small_counts_get_fisher(self):
        big = pd.DataFrame({"f": [1] * 480 + [0] * 20 + [1] * 40 + [0] * 460, "g": ["a"] * 500 + ["b"] * 500})
        r, _ = captured(compare_proportions, big, "f", "g", "a", "b")
        self.assertTrue(0 < r.p < 1e-100)
        small = pd.DataFrame({"ok": [True] + [False] * 9 + [True] * 6 + [False] * 4, "g": ["x"] * 10 + ["y"] * 10})
        self.assertIsNotNone(captured(compare_proportions, small, "ok", "g", "x", "y")[0].fisher_p)

    def monthly_frame(self):
        rng = np.random.default_rng(8)
        days = pd.date_range("2023-01-01", "2024-06-30", freq="D")
        return pd.DataFrame({"d": days, "v": rng.uniform(10, 100, len(days)) * (1 + (days.month == 12))})

    def test_trend_forecast_matches_hand_computation(self):
        df = self.monthly_frame()
        monthly = df.groupby(df.d.dt.strftime("%Y-%m")).v.sum()
        slope, intercept = np.polyfit(np.arange(len(monthly)), monthly.to_numpy(), 1)
        r, _ = captured(trend_forecast, df, "d", "v")
        self.assertAlmostEqual(r.slope, slope, places=6)
        self.assertAlmostEqual(r.forecast, intercept + slope * len(monthly), places=6)
        self.assertEqual((r.best, r.worst, r.n_periods), (monthly.idxmax(), monthly.idxmin(), len(monthly)))
        self.assertEqual(list(r.series["index"]), list(range(len(monthly))))

    def test_trend_forecast_gaps_frequencies_and_errors(self):
        df = self.monthly_frame()
        gap = df[~df.d.dt.strftime("%Y-%m").isin(["2023-05", "2023-06"])]
        r, _ = captured(trend_forecast, gap, "d", "v")
        self.assertEqual(r.n_periods, 18)  # missing months keep their place
        self.assertEqual(r.series.loc[r.series.period == "2023-05", "value"].iloc[0], 0)
        self.assertEqual(len(captured(trend_forecast, df, "d", "v", freq="Q", periods=2)[0].forecast_table), 2)
        as_text = df.assign(d=df.d.dt.strftime("%Y-%m-%d"))
        self.assertAlmostEqual(captured(trend_forecast, as_text, "d", "v")[0].slope, captured(trend_forecast, df, "d", "v")[0].slope, places=9)
        with self.assertRaisesRegex(ValueError, "freq must"):
            trend_forecast(df, "d", "v", freq="X")
        with self.assertRaisesRegex(ValueError, "at least 3"):
            trend_forecast(df.head(5), "d", "v", freq="Y")
        with self.assertRaisesRegex(ValueError, "dates"):
            trend_forecast(df.assign(d="not a date"), "d", "v")

    def test_top_correlations_and_charts(self):
        rng = np.random.default_rng(9)
        df = pd.DataFrame({"x": rng.normal(size=300), "z": rng.normal(size=300), "g": rng.choice(list("ab"), 300), "h": rng.choice(list("pqr"), 300)})
        df["y"] = 2 * df.x + rng.normal(size=300) * 0.1
        t = captured(top_correlations, df, n=2)[0].table
        self.assertEqual({t.iloc[0].var1, t.iloc[0].var2}, {"x", "y"})
        for i, (kind, kw) in enumerate([("bar", dict(x="g", y="x")), ("line", dict(x="z", y="x", agg="mean")), ("hist", dict(x="x")),
                                        ("scatter", dict(x="x", y="y", hue="g")), ("box", dict(x="g", y="y")), ("bar", dict(x="g", y="y", hue="h"))]):
            path = captured(quick_chart, df, kind=kind, path=f"c{i}.png", **kw)[0]
            self.assertGreater((self.tmp / path).stat().st_size, 3000, kind)
        with self.assertRaisesRegex(ValueError, "kind must"):
            quick_chart(df, x="x", kind="pie3d")
        with self.assertRaisesRegex(ValueError, "needs y"):
            quick_chart(df, x="x", kind="scatter")


class ExcelTests(InTempDir):
    def messy_workbook(self, name="messy.xlsx"):
        from openpyxl import Workbook

        rng = np.random.default_rng(7)
        wb = Workbook()
        ws = wb.active
        ws.title = "Summary"
        ws["A1"] = "Q3 Sales Report - CONFIDENTIAL"
        ws.merge_cells("A1:E1")
        ws["A2"] = "Prepared by finance"
        ws.append([])
        ws.append(["Region", "Product", "Units", "Price", "Revenue"])
        data = []
        for i in range(40):
            u, p = int(rng.integers(1, 20)), round(float(rng.uniform(5, 50)), 2)
            data.append(["NSEW"[i % 4], "ABC"[i % 3], u, p, round(u * p, 2)])
            ws.append(data[-1])
        ws.append(["Total", None, "=SUM(C5:C44)", None, "=SUM(E5:E44)"])
        ws2 = wb.create_sheet("Regional")
        ws2.append(["Region", "Month", "Revenue"])
        r = 2
        for reg, vals in {"North": [100, 150, 200], "South": [80, 90, 70], "West": [300, 250, 400]}.items():
            for m, v in zip(["Jan", "Feb", "Mar"], vals):
                ws2.append([reg if m == "Jan" else None, m, v])
            ws2.merge_cells(start_row=r, start_column=1, end_row=r + 2, end_column=1)
            r += 3
        ws2.append(["Grand total", None, 1190])
        wb.create_sheet("Notes")["A1"] = "just a note"
        wb.save(name)
        return round(sum(d[4] for d in data), 2)

    def test_inspect_and_read_messy_workbook(self):
        revenue = self.messy_workbook()
        info, _ = captured(inspect_workbook, "messy.xlsx")
        sheets = {s["name"]: s for s in info.sheets}
        self.assertEqual((sheets["Summary"]["header_row"], sheets["Summary"]["table_rows"], sheets["Summary"]["total_rows"]), (4, 40, 1))
        self.assertEqual(sheets["Summary"]["formulas_without_value"], 2)
        self.assertEqual((sheets["Regional"]["merged"], sheets["Regional"]["header_row"]), (3, 1))
        df, out = captured(read_excel_smart, "messy.xlsx", "Summary")
        self.assertEqual(list(df.columns), ["Region", "Product", "Units", "Price", "Revenue"])
        self.assertEqual(len(df), 40)
        self.assertAlmostEqual(df.Revenue.sum(), revenue, places=2)
        self.assertIn("never calculated", out)
        reg = read_excel_smart("messy.xlsx", "Regional", verbose=False)  # merged cells filled down, Grand total dropped
        self.assertEqual(reg.groupby("Region").Revenue.sum().to_dict(), {"North": 450, "South": 240, "West": 950})

    def test_sheet_selection_aliases_and_errors(self):
        self.messy_workbook()
        a = read_excel_smart("messy.xlsx", sheet_name="Regional", verbose=False)
        b = read_excel_smart("messy.xlsx", sheet="Regional", verbose=False)
        self.assertTrue(a.equals(b))
        self.assertEqual(read_excel_smart("messy.xlsx", verbose=False).shape, (40, 5))  # default = biggest sheet
        self.assertEqual(read_excel_smart("messy.xlsx", 1, verbose=False).shape, a.shape)
        with self.assertRaisesRegex(KeyError, "Sheets:"):
            read_excel_smart("messy.xlsx", "Nope")
        with self.assertRaisesRegex(TypeError, "It accepts"):
            read_excel_smart("messy.xlsx", sheet_nam="Regional")
        with self.assertRaisesRegex(TypeError, "not both"):
            read_excel_smart("messy.xlsx", sheet="Summary", sheet_name="Regional")
        with self.assertRaisesRegex(TypeError, "It accepts"):
            inspect_workbook("messy.xlsx", sheet_name="x")

    def sales(self):
        return pd.DataFrame({"Country": ["US", "UK", "France", "Germany"], "Revenue": [27975547, 10646196, 8432872, 8978596],
                             "Profit": [9e6, 3.1e6, 2.2e6, 2.9e6]})

    def test_save_excel_totals_formulas_and_charts(self):
        from openpyxl import load_workbook

        sales = self.sales()
        captured(save_excel, "out.xlsx", {"Summary": sales, "Other": sales.head(2)}, totals=True, formulas={"Margin": ("{Profit}/{Revenue}", "0.0%")},
                 charts=[{"kind": "bar", "x": "Country", "y": "Revenue", "title": "Revenue by country"},
                         {"kind": "line", "x": "Country", "y": ["Revenue", "Profit"], "anchor": "H20"},
                         {"kind": "pie", "x": "Country", "y": "Revenue", "sheet": "Other"}])
        wb = load_workbook("out.xlsx")
        s = wb["Summary"]
        self.assertEqual((s["D1"].value, s["D2"].value, s["D2"].number_format, s["D5"].value), ("Margin", "=C2/B2", "0.0%", "=C5/B5"))
        self.assertEqual((s["A6"].value, s["B6"].value, s["C6"].value), ("Total", "=SUM(B2:B5)", "=SUM(C2:C5)"))
        self.assertEqual((len(s._charts), len(wb["Other"]._charts), s.auto_filter.ref), (2, 1, "A1:D5"))

    def test_formula_templates_are_resolved_and_validated(self):
        from openpyxl import load_workbook

        df = pd.DataFrame({"Price": [2.0, 4.0], "Unit Price": [3.0, 5.0], "Qty": [1, 2]})
        captured(save_excel, "a.xlsx", df, formulas={"T": "=Unit Price*Qty+Price"})  # bare names, longest first
        self.assertEqual(load_workbook("a.xlsx")["Data"]["D2"].value, "=B2*C2+A2")
        captured(save_excel, "b.xlsx", df, formulas={"F": ("IF({Qty}>0,{Price}/{Qty},0)", "0.0%"), "G": "=ROUND(Price/Qty*100,1)"})
        w = load_workbook("b.xlsx")["Data"]
        self.assertEqual((w["D2"].value, w["E2"].value), ("=IF(C2>0,A2/C2,0)", "=ROUND(A2/C2*100,1)"))
        for formulas, pattern in (({"Price": "{Qty}"}, "already exists"), ({"Z": "=E2/C2"}, "mentions no column"),
                                  ({"Z": "=Revnue/Qty"}, "unknown name"), ({"Z": "{Nope}/{Qty}"}, "unknown columns")):
            with self.assertRaisesRegex(ValueError, pattern):
                save_excel("x.xlsx", df, formulas=formulas)

    def test_sheet_names_formats_and_chart_errors(self):
        from openpyxl import load_workbook

        df = pd.DataFrame({"Year": [2011, 2012], "Units": [1500, 2500], "Rate": [0.5, 0.25], "Big": [9e6, 8.2e6]})
        captured(save_excel, "s.xlsx", df, sheet_name="Summary", totals=True, formats={"Rate": "0%"})
        ws = load_workbook("s.xlsx")["Summary"]
        self.assertEqual((ws["A2"].number_format, ws["B2"].number_format, ws["C2"].number_format), ("0", "#,##0", "0%"))  # 2011, not 2,011
        self.assertGreaterEqual(ws.column_dimensions["D"].width, len("17,200,000.00") + 2)  # the Total row must fit
        captured(save_excel, "t.xlsx", {"Q3": df})
        self.assertEqual(load_workbook("t.xlsx").sheetnames, ["Q3"])
        captured(save_excel, "d.xlsx", df)
        self.assertEqual(load_workbook("d.xlsx").sheetnames, ["Data"])
        with self.assertRaisesRegex(ValueError, "not found"):
            save_excel("x.xlsx", df, charts=[{"x": "Nope", "y": "Units"}])
        with self.assertRaisesRegex(ValueError, "aggregate first"):
            save_excel("x.xlsx", pd.DataFrame({"a": range(100), "b": range(100)}), charts=[{"x": "a", "y": "b"}])

    @unittest.skipUnless(shutil.which("soffice"), "LibreOffice not installed")
    def test_formulas_calculate_correctly_in_a_real_spreadsheet_engine(self):
        from openpyxl import load_workbook

        sales = self.sales()
        captured(save_excel, "calc.xlsx", sales, totals=True, formulas={"Margin": "{Profit}/{Revenue}"})
        subprocess.run(["soffice", "--headless", "--convert-to", "xlsx", "--outdir", "recalc", "calc.xlsx"], capture_output=True, timeout=180)
        ws = load_workbook("recalc/calc.xlsx", data_only=True)["Data"]
        self.assertAlmostEqual(ws["B6"].value, sales.Revenue.sum(), places=6)
        self.assertAlmostEqual(ws["C6"].value, sales.Profit.sum(), places=6)
        self.assertAlmostEqual(ws["D2"].value, sales.Profit[0] / sales.Revenue[0], places=9)


if __name__ == "__main__":
    unittest.main()
