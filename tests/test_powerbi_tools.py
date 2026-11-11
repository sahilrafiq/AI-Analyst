"""Tests for powerbi_tools.py: date table, star schema (lossless round trip from disk), M, DAX and expected values."""
import math
import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from _common import InTempDir, captured
from analyst_tools import clean_columns
from powerbi_tools import (build_powerbi_package, check_dax, dax_measures, export_star_schema, expected_values, m_load_csv, m_load_excel,
                           make_date_table)


class DateTableTests(unittest.TestCase):
    def test_continuous_with_calendar_columns(self):
        dt = make_date_table("2023-11-15", "2024-06-30")
        self.assertEqual(len(dt), 229)
        self.assertTrue(dt.Date.diff().dropna().eq(pd.Timedelta(days=1)).all())
        row = dt.set_index("Date").loc["2024-02-29"]
        self.assertEqual((row.Weekday, row.MonthNumber, row.YearMonth, row.YearMonthSort, bool(row.IsWeekend)), ("Thu", 2, "2024-02", 202402, False))
        self.assertTrue(dt.set_index("Date").loc["2024-03-02", "IsWeekend"])

    def test_fiscal_year_and_errors(self):
        fy = make_date_table("2024-01-01", "2024-12-31", fiscal_start_month=4).set_index("Date")
        self.assertEqual((fy.loc["2024-01-15", "FiscalYear"], fy.loc["2024-04-01", "FiscalYear"]), (2024, 2025))
        self.assertEqual((fy.loc["2024-04-01", "FiscalQuarter"], fy.loc["2024-01-15", "FiscalQuarter"]), (1, 4))
        with self.assertRaises(ValueError):
            make_date_table("2024-02-01", "2024-01-01")
        with self.assertRaises(ValueError):
            make_date_table("2024-01-01", "2024-12-31", fiscal_start_month=13)


def flat_sales(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    cat = {"Bike": ["Road", "Mountain"], "Gear": ["Helmet", "Lock", "Tire"]}
    rows = []
    for _ in range(n):
        c = rng.choice(list(cat))
        s = rng.choice(cat[c])
        rows.append((pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(rng.integers(0, 900))), c, s, f"{s}-{int(rng.integers(1, 4))}",
                     rng.choice(["US", "UK", "FR"]), rng.choice(["F", "M", None]), int(rng.integers(1, 9)), round(float(rng.uniform(5, 90)), 2)))
    return pd.DataFrame(rows, columns=["Date", "Category", "SubCategory", "Product", "Country", "Gender", "Qty", "Revenue"])


DIMS = {"Product": ["Product", "SubCategory", "Category"], "Customer": ["Gender"], "Location": ["Country"]}


class StarSchemaTests(InTempDir):
    def test_split_is_lossless_when_read_back_from_the_csv_files(self):
        flat = flat_sales()
        star, _ = captured(export_star_schema, flat, "pbi", "Sales", "Date", DIMS)
        self.assertTrue(all(star.checks.values()), star.checks)
        f = pd.read_csv("pbi/Sales.csv", parse_dates=["Date"])
        self.assertEqual((len(f), list(f.columns[:4])), (2000, ["Date", "ProductKey", "CustomerKey", "LocationKey"]))
        back = (f.merge(pd.read_csv("pbi/Product.csv"), on="ProductKey").merge(pd.read_csv("pbi/Customer.csv", keep_default_na=False), on="CustomerKey")
                .merge(pd.read_csv("pbi/Location.csv"), on="LocationKey"))
        keys = ["Date", "Product", "Revenue", "Qty"]
        orig = flat.assign(Gender=flat.Gender.fillna("(Unknown)")).sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(back[orig.columns].sort_values(keys).reset_index(drop=True), orig, check_dtype=False)
        self.assertIn("(Unknown)", pd.read_csv("pbi/Customer.csv", keep_default_na=False).Gender.values)  # missing text becomes a member
        dd = pd.read_csv("pbi/Date.csv", parse_dates=["Date"])  # a complete calendar year range, no gaps
        self.assertEqual((dd.Date.min(), dd.Date.max(), len(dd)), (pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-31"), 1096))
        self.assertAlmostEqual(f.Revenue.sum(), flat.Revenue.sum(), places=6)

    def test_all_expected_files_are_written(self):
        captured(export_star_schema, flat_sales(), "pbi", "Sales", "Date", DIMS)
        for name in ("Sales", "Product", "Customer", "Location", "Date"):
            self.assertTrue(os.path.exists(f"pbi/{name}.csv"), name)
        for name in ("load_queries.m", "relationships.md", "data_dictionary.csv"):
            self.assertTrue(os.path.exists(f"pbi/{name}"), name)
        self.assertIn("Sales[ProductKey] | Product[ProductKey]", Path("pbi/relationships.md").read_text(encoding="utf-8"))

    def test_bad_input_gets_clear_errors(self):
        flat = flat_sales(300)
        for dims, exc, pattern in ((dict(dims={"A": ["Product"], "B": ["Product"]}), ValueError, "more than one dimension"),
                                   (dict(dims={"A": ["Date"]}), ValueError, "date column"), (dict(dims={"A": ["Nope"]}), KeyError, "not found")):
            with self.assertRaisesRegex(exc, pattern):
                export_star_schema(flat, "x", "Sales", "Date", **dims)
        with self.assertRaisesRegex(ValueError, "no valid date"):
            export_star_schema(flat.assign(Date=None), "x")

    def test_dimensions_can_be_detected_automatically(self):
        star, _ = captured(export_star_schema, flat_sales(500), "auto", "Sales", "Date")
        self.assertTrue(star.notes)
        self.assertTrue({"Category", "SubCategory", "Country", "Gender"} <= set(star.tables))

    def test_power_query_scripts_are_well_formed_and_typed(self):
        captured(export_star_schema, flat_sales(500), "pbi", "Sales", "Date", DIMS)
        m = Path("pbi/load_queries.m").read_text(encoding="utf-8")
        self.assertEqual((m.count("let\n"), m.count("\nin\n"), m.count("Csv.Document")), (5, 5, 5))
        for chunk in m.split("// Query name:")[1:]:
            self.assertEqual((chunk.count("("), chunk.count("{"), chunk.count('"') % 2), (chunk.count(")"), chunk.count("}"), 0))
        sales = m.split("// Query name: Sales")[1].split("// Query name:")[0]
        for fragment in ('{"Date", type date}', '{"ProductKey", Int64.Type}', '{"Revenue", type number}'):
            self.assertIn(fragment, sales)

    def test_excel_query_skips_title_rows_fills_merged_cells_and_drops_totals(self):
        df = pd.DataFrame({"Region": ["North", "South"], "Units": [3, 4], "Revenue": [10.5, 20.0]})
        q = m_load_excel("Summary", "m.xlsx", "Summary", df, header_row=4, fill_down=["Region"])
        for fragment in ("Table.Skip(SheetData, 3)", 'Table.FillDown(Promoted, {"Region"})', 'Record.Field(_, "Region")', '{"Revenue", type number}'):
            self.assertIn(fragment, q)
        self.assertEqual((q.count("("), q.count("{"), q.count('"') % 2), (q.count(")"), q.count("}"), 0))
        self.assertTrue(q.startswith("// Query name: Summary\nlet") and q.rstrip().endswith("Typed"))
        self.assertIn("Csv.Document", m_load_csv("T", "t.csv", df))

    def sales_with_date_parts(self, with_parts=True):
        rng = np.random.default_rng(3)
        n = 5000
        true = pd.Timestamp("2011-01-01") + pd.to_timedelta(rng.integers(0, 2040, n), unit="D")
        s = pd.DataFrame({"Date": true.strftime("%m/%d/%Y"), "Product": rng.choice(list("ABCDEFGH"), n), "Category": "Bikes",
                          "Revenue": np.round(rng.uniform(5, 900, n), 2)})
        s.loc[s.Product.isin(list("EFGH")), "Category"] = "Gear"
        s.loc[(s.Product == "A") & (rng.random(n) < 0.2), "Category"] = "Gear"  # product A under two categories
        if with_parts:
            s["Day"], s["Month"], s["Year"] = true.day, true.month_name(), true.year
        return pd.concat([s, s.head(40)], ignore_index=True)  # 40 exact duplicates

    def export(self, frame, out):
        return captured(export_star_schema, frame, out, "Sales", "Date", {"Product": ["Product", "Category"]})

    def test_notes_for_duplicates_and_products_under_two_categories(self):
        good, _ = captured(clean_columns, self.sales_with_date_parts())
        star, _ = self.export(good, "ok")
        self.assertTrue(star.checks["date_agrees_with_year_month_day_columns"])
        self.assertFalse(any("WARNING" in n for n in star.notes))
        self.assertTrue(any("40 exact duplicate rows" in n for n in star.notes))
        self.assertTrue(any("distinct 'Product' values" in n for n in star.notes))

    def test_swapped_day_and_month_is_detected_from_year_month_day_columns(self):
        sales = self.sales_with_date_parts()
        good, _ = captured(clean_columns, sales)
        wrong, log = captured(clean_columns, sales, dayfirst=True)
        self.assertIn("WARNING", log)  # the explicit wrong choice is already flagged by clean_columns
        star, printed = self.export(wrong, "bad")
        self.assertIs(star.checks["date_agrees_with_year_month_day_columns"], False)
        self.assertTrue(any("probably swapped" in n for n in star.notes))
        self.assertIn("FAILED", printed)
        self.assertFalse(np.allclose(good.groupby(good.Date.dt.month).Revenue.sum().values, wrong.groupby(wrong.Date.dt.month).Revenue.sum().values))
        plain, _ = captured(clean_columns, self.sales_with_date_parts(with_parts=False))
        self.assertNotIn("date_agrees_with_year_month_day_columns", self.export(plain, "plain")[0].checks)


class DaxTests(unittest.TestCase):
    def test_measure_templates(self):
        dx = captured(dax_measures, "Sales", "Revenue", rank_by=("Product", "Product"),
                      tables={"Sales": ["Revenue"], "Date": ["Date", "YearMonth"], "Product": ["Product"]})[0]
        self.assertEqual(dx.problems, [])
        self.assertEqual(dx.measures["Total Revenue"], "SUM('Sales'[Revenue])")
        self.assertIn("SAMEPERIODLASTYEAR('Date'[Date])", dx.measures["Revenue LY"])
        self.assertEqual(dx.measures["Revenue YTD"], "TOTALYTD([Total Revenue], 'Date'[Date])")
        self.assertIn("RANKX(ALL('Product'[Product])", dx.measures["Revenue Rank by Product"])
        self.assertIn("VAR _window", dx.text)
        self.assertEqual(len(dx.measures), 13)

    def test_options_and_awkward_names(self):
        fy = captured(dax_measures, "Sales", "Revenue", fiscal_year_end="3/31", which=["ytd"])[0]
        self.assertTrue(fy.measures["Revenue YTD"].endswith('"3/31")'))
        self.assertEqual(set(fy.measures), {"Total Revenue", "Revenue YTD"})
        self.assertEqual(set(captured(dax_measures, which=["yoy"])[0].measures), {"Total Revenue", "Revenue LY", "Revenue YoY %"})
        weird = captured(dax_measures, "My Sales", "Net $ [adj]", label="Net")[0]
        self.assertEqual(weird.problems, [])
        self.assertIn("'My Sales'[Net $ [adj]]]", weird.measures["Total Net"])  # ] is escaped as ]]
        with self.assertRaisesRegex(ValueError, "choose from"):
            dax_measures(which=["bogus"])

    def test_structure_checker(self):
        self.assertTrue(check_dax({"A": "SUM('S'[x]"}))  # unbalanced parenthesis
        self.assertTrue(check_dax({"A": "[Missing] + 1"}))  # undefined measure
        self.assertTrue(check_dax({"A": "SUM('S'[nope])"}, {"S": ["x"]}))  # unknown column
        self.assertTrue(check_dax({"A": "SUM('S[x])"}))  # unbalanced quote
        self.assertEqual(check_dax({"A": "SUM('S'[x])", "B": "[A] * 2"}, {"S": ["x"]}), [])


def reference_values(totals, fiscal_start):
    """Plain-Python definition of every expected measure, used to check expected_values()."""
    keys = sorted(totals)
    allm = [(y, m) for y in range(keys[0][0], keys[-1][0] + 1) for m in range(1, 13)]
    rng = [k for k in allm if keys[0] <= k <= keys[-1]]
    back = lambda k, d: (lambda t: (t // 12, t % 12 + 1))(k[0] * 12 + k[1] - 1 - d)
    nan = float("nan")
    out = []
    for k in rng:
        t, ly, pv = totals.get(k, nan), totals.get(back(k, 12), nan), totals.get(back(k, 1), nan)
        fyr = lambda x: x[0] + (1 if fiscal_start > 1 and x[1] >= fiscal_start else 0)
        win = sum(totals.get(back(k, j), 0) for j in range(3))
        divisor = sum(1 for j in range(3) if back(k, j) >= (rng[0][0], 1))
        out.append(dict(t=t, ly=ly, yoy=(t - ly) / ly if ly == ly and ly else nan, pv=pv, mom=(t - pv) / pv if pv == pv and pv else nan,
                        ytd=sum(totals.get(x, 0) for x in rng if fyr(x) == fyr(k) and x <= k), run=sum(totals.get(x, 0) for x in rng if x <= k),
                        mov=win / divisor, pct=t / sum(totals.values())))
    return out


class ExpectedValuesTests(InTempDir):
    def data(self):
        months = [(2023 + i // 12, i % 12 + 1) for i in range(23)]  # 2023-01 .. 2024-11; July 2023 has no sales
        rows, totals = [], {}
        for i, (y, m) in enumerate(months):
            if (y, m) == (2023, 7):
                continue
            v = 100 + 10 * i + (i % 3) * 5
            rows += [(pd.Timestamp(y, m, 3), "A", v * 0.4), (pd.Timestamp(y, m, 17), "B", v * 0.6)]
            totals[(y, m)] = float(v)
        return pd.DataFrame(rows, columns=["Date", "Cat", "Revenue"]), totals

    def test_every_column_matches_the_plain_python_reference(self):
        df, totals = self.data()
        for fiscal_start in (1, 4):
            ev, _ = captured(expected_values, df, "Date", "Revenue", by="Cat", fiscal_start_month=fiscal_start)
            ref = reference_values(totals, fiscal_start)
            self.assertEqual(len(ev.month), len(ref))
            for col, key in (("Total Revenue", "t"), ("Revenue LY", "ly"), ("Revenue YoY %", "yoy"), ("Revenue YTD", "ytd"), ("Revenue Previous Month", "pv"),
                             ("Revenue MoM %", "mom"), ("Revenue Running Total", "run"), ("Revenue 3M Moving Avg", "mov"), ("Revenue % of Total", "pct")):
                np.testing.assert_allclose(ev.month[col].to_numpy(float), np.array([r[key] for r in ref], float), rtol=1e-9, atol=1e-9, equal_nan=True,
                                           err_msg=f"{col} (fiscal start {fiscal_start})")

    def test_year_month_and_category_details(self):
        df, totals = self.data()
        ev, _ = captured(expected_values, df, "Date", "Revenue", by="Cat")
        y = ev.year.set_index("Year")
        self.assertAlmostEqual(y.loc[2023, "Total Revenue"], sum(v for k, v in totals.items() if k[0] == 2023))
        self.assertTrue(math.isnan(y.loc[2023, "Revenue LY"]))
        self.assertAlmostEqual(y.loc[2024, "Revenue YoY %"], y.loc[2024, "Total Revenue"] / y.loc[2023, "Total Revenue"] - 1)
        m = ev.month.set_index("YearMonth")
        self.assertAlmostEqual(m.loc["2023-01", "Revenue 3M Moving Avg"], totals[(2023, 1)] / 1)  # only one month exists in the date table
        self.assertAlmostEqual(m.loc["2023-02", "Revenue 3M Moving Avg"], (totals[(2023, 1)] + totals[(2023, 2)]) / 2)
        self.assertTrue(math.isnan(m.loc["2023-07", "Total Revenue"]))  # a month with no sales is blank...
        self.assertGreater(m.loc["2023-07", "Revenue YTD"], 0)  # ...but the year-to-date carries on
        self.assertEqual((list(ev.by["Cat"]), list(ev.by["Rank"])), (["B", "A"], [1, 2]))
        self.assertAlmostEqual(ev.by["Revenue % of Total"].sum(), 1)


class PackageTests(InTempDir):
    def test_package_files_fiscal_year_end_and_excel_formats(self):
        from openpyxl import load_workbook

        flat = pd.DataFrame({"Date": pd.date_range("2022-01-01", periods=400, freq="D"), "Category": ["x", "y"] * 200, "Product": list("abcd") * 100,
                             "Revenue": np.arange(400) * 1.5})
        pkg, _ = captured(build_powerbi_package, flat, "pkg", fact="Sales", date="Date", value="Revenue", dims={"Product": ["Product", "Category"]},
                          fiscal_start_month=4, rank_by=("Product", "Product"))
        names = set(os.listdir("pkg"))
        self.assertTrue({"Sales.csv", "Product.csv", "Date.csv", "load_queries.m", "relationships.md", "data_dictionary.csv", "measures.dax",
                         "expected_values.xlsx", "README_PowerBI.md"} <= names)
        self.assertIn("TOTALYTD([Total Revenue], 'Date'[Date], \"3/31\")", Path("pkg/measures.dax").read_text(encoding="utf-8"))
        wb = load_workbook("pkg/expected_values.xlsx")
        self.assertEqual(wb.sheetnames, ["By Year", "By Month", "By Product"])
        header = [c.value for c in wb["By Year"][1]]
        self.assertEqual(wb["By Year"]["A2"].number_format, "0")  # 2011, not 2,011
        self.assertEqual(wb["By Year"].cell(3, header.index("Revenue YoY %") + 1).number_format, "0.0%")
        self.assertIn("expected_values.xlsx", Path("pkg/README_PowerBI.md").read_text(encoding="utf-8"))
        self.assertTrue(all(pkg.star.checks.values()))

    def test_value_column_cannot_also_be_a_dimension(self):
        flat = pd.DataFrame({"Date": pd.date_range("2022-01-01", periods=50), "Revenue": range(50)})
        with self.assertRaisesRegex(ValueError, "measure column"):
            build_powerbi_package(flat, "x", value="Revenue", dims={"P": ["Revenue"]})


if __name__ == "__main__":
    unittest.main()
