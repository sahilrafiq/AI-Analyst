"""
powerbi_tools.py - prepare data for Power BI Desktop and write Power Query (M) / DAX for it.

Power BI cannot be driven from Python, and DAX/M cannot be executed here. So this module does the parts
that can be made reliable:
  * a clean star schema (fact + dimension tables + a complete Date table) as CSV files,
  * Power Query (M) load scripts with the right column types,
  * DAX measures from tested templates (YoY, YTD, running total, ...), checked for structure,
  * expected_values.xlsx computed independently in pandas, so you can compare it with the numbers
    your Power BI visuals show and catch any mistake in the model or the measures.

    from powerbi_tools import *
    pkg = build_powerbi_package(df, "powerbi_out", fact="Sales", date="Date", value="Revenue",
                                dims={"Product": ["Product", "Sub_Category", "Product_Category"],
                                      "Customer": ["Customer_Gender", "Age_Group"], "Location": ["Country", "State"]})
Then open powerbi_out/README_PowerBI.md and follow the steps.
"""
import calendar
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analyst_tools import Result, _col

__all__ = [
    "make_date_table",
    "export_star_schema",
    "m_load_csv",
    "m_load_excel",
    "dax_measures",
    "check_dax",
    "expected_values",
    "build_powerbi_package",
]

def _is_text(s):
    """Text column in any pandas version (object in pandas 2, str in pandas 3)."""
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ------------------------------------------------------------------ date table
def make_date_table(start, end, fiscal_start_month=1):
    """A complete, gap-free Date table (one row per day) with year/quarter/month/week-day columns and fiscal year."""
    d = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")
    if len(d) == 0:
        raise ValueError("end must not be before start")
    fm = int(fiscal_start_month)
    if not 1 <= fm <= 12:
        raise ValueError("fiscal_start_month must be 1-12")
    t = pd.DataFrame({"Date": d})
    t["Year"] = np.asarray(d.year)
    t["QuarterNumber"] = np.asarray(d.quarter)
    t["Quarter"] = "Q" + t["QuarterNumber"].astype(str)
    t["MonthNumber"] = np.asarray(d.month)
    t["Month"] = [_MONTHS[m - 1] for m in d.month]
    t["YearMonth"] = d.strftime("%Y-%m")
    t["YearMonthSort"] = np.asarray(d.year) * 100 + np.asarray(d.month)
    t["Day"] = np.asarray(d.day)
    t["WeekdayNumber"] = np.asarray(d.weekday) + 1
    t["Weekday"] = [_DAYS[w] for w in d.weekday]
    t["IsWeekend"] = np.asarray(d.weekday) >= 5
    t["FiscalYear"] = np.asarray(d.year) + ((np.asarray(d.month) >= fm) & (fm > 1)).astype(int)
    t["FiscalQuarter"] = ((np.asarray(d.month) - fm) % 12) // 3 + 1
    return t


# ------------------------------------------------------------ Power Query (M)
def _mq(text):
    return '"' + str(text).replace('"', '""') + '"'


def _m_type(s):
    if pd.api.types.is_bool_dtype(s):
        return "type logical"
    if pd.api.types.is_integer_dtype(s):
        return "Int64.Type"
    if pd.api.types.is_float_dtype(s):
        return "type number"
    if pd.api.types.is_datetime64_any_dtype(s):
        v = s.dropna()
        return "type date" if (v.dt.normalize() == v).all() else "type datetime"
    return "type text"


def _m_types(df):
    return ", ".join("{" + f"{_mq(c)}, {_m_type(df[c])}" + "}" for c in df.columns)


def m_load_csv(table, path, df):
    """Power Query (M) that loads a CSV written by export_star_schema and sets every column type."""
    return (
        f"// Query name: {table}\nlet\n"
        f"    Source = Csv.Document(File.Contents({_mq(Path(path).resolve())}), [Delimiter=\",\", Columns={len(df.columns)}, Encoding=65001, QuoteStyle=QuoteStyle.Csv]),\n"
        f"    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),\n"
        f"    Typed = Table.TransformColumnTypes(Promoted, {{{_m_types(df)}}}, \"en-US\")\n"
        f"in\n    Typed"
    )


def m_load_excel(table, path, sheet, df, header_row=1, drop_total_rows=True, fill_down=None):
    """Power Query (M) for a messy Excel sheet: skips title rows, promotes the header, optionally fills merged
    cells down, drops Total/Subtotal rows (looks at the first column) and sets column types from `df`.
    header_row = Excel row number of the header (read_excel_smart prints it)."""
    first = df.columns[0]
    steps = [
        f"    Source = Excel.Workbook(File.Contents({_mq(Path(path).resolve())}), null, true),",
        f"    SheetData = Source{{[Item={_mq(sheet)},Kind=\"Sheet\"]}}[Data],",
        f"    Skipped = Table.Skip(SheetData, {max(int(header_row) - 1, 0)}),",
        "    Promoted = Table.PromoteHeaders(Skipped, [PromoteAllScalars=true]),",
    ]
    last = "Promoted"
    if fill_down:
        steps.append(f"    Filled = Table.FillDown({last}, {{{', '.join(_mq(c) for c in fill_down)}}}),")
        last = "Filled"
    if drop_total_rows:
        steps.append(
            f"    NoTotals = Table.SelectRows({last}, each not (try List.AnyTrue(List.Transform({{\"total\", \"grand total\", \"subtotal\", \"sub-total\"}}, "
            f"(p) => Text.StartsWith(Text.Lower(Text.From(Record.Field(_, {_mq(first)}))), p))) otherwise false)),"
        )
        last = "NoTotals"
    steps.append(f"    Typed = Table.TransformColumnTypes({last}, {{{_m_types(df)}}}, \"en-US\")")
    return f"// Query name: {table}\nlet\n" + "\n".join(steps) + "\nin\n    Typed"


def _date_part_check(work, date):
    """Share of rows where the parsed date agrees with columns named Year / Month / Day (if the data has them).
    This catches a swapped day/month order, which silently moves revenue into the wrong months."""
    lower = {str(c).lower(): c for c in work.columns if str(c).lower() in ("year", "month", "day")}
    d, shares = work[date], {}
    if "year" in lower and pd.api.types.is_numeric_dtype(work[lower["year"]]):
        shares["Year"] = float((work[lower["year"]].astype(float) == d.dt.year).mean())
    if "day" in lower and pd.api.types.is_numeric_dtype(work[lower["day"]]):
        shares["Day"] = float((work[lower["day"]].astype(float) == d.dt.day).mean())
    if "month" in lower:
        col = work[lower["month"]]
        if pd.api.types.is_numeric_dtype(col):
            shares["Month"] = float((col.astype(float) == d.dt.month).mean())
        else:
            names = d.dt.month.map(lambda m: _MONTHS[m - 1].lower())
            shares["Month"] = float((col.astype(str).str.strip().str[:3].str.lower() == names).mean())
    return shares


# ----------------------------------------------------------------- star schema
def export_star_schema(df, outdir, fact="Sales", date="Date", dims=None, fiscal_start_month=1, verify=True):
    """Split a flat table into a fact table + dimension tables with integer keys + a complete Date table,
    and write them as CSV together with load_queries.m (Power Query), relationships.md and data_dictionary.csv.
    dims = {'Product': ['Product', 'Sub_Category', 'Product_Category'], 'Location': ['Country', 'State']}:
    each dimension holds the listed columns (one row per distinct combination) and the fact gets a ProductKey.
    Without dims, every text column with 2-50 values becomes its own dimension.
    Result keys: tables, files, relationships, checks, notes, outdir."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    date = _col(df, date, "date")
    work = df.copy()
    work[date] = pd.to_datetime(work[date]).dt.normalize()
    if work[date].isna().any():
        raise ValueError(f"{int(work[date].isna().sum())} rows have no valid date in '{date}'; fix or drop them first")
    notes = []
    n_dup = int(df.duplicated().sum())
    if n_dup:
        notes.append(f"{n_dup:,} exact duplicate rows in the input are kept in the fact table (every total includes them); remove them first with df.drop_duplicates() if they are errors")
    part_shares = _date_part_check(work, date)
    bad_parts = {k: v for k, v in part_shares.items() if v < 0.99}
    if bad_parts:
        notes.append("WARNING: the parsed '" + date + "' disagrees with the " + ", ".join(f"{k} column ({v:.1%} agree)" for k, v in bad_parts.items())
                     + ". The day/month order is probably swapped, which puts revenue into the wrong months: parse the dates again with the other order (clean_columns(df, dayfirst=True or False))")
    if dims is None:
        dims = {c: [c] for c in work.columns if c != date and _is_text(work[c]) and 2 <= work[c].nunique() <= 50}
        notes.append("dims were detected automatically (one per text column with 2-50 values); pass dims={'Product': ['Product', 'Category']} to group columns")
    dims = {str(k): [_col(work, c, "dims") for c in v] for k, v in dims.items()}
    used = [c for cols in dims.values() for c in cols]
    if date in used:
        raise ValueError(f"the date column '{date}' cannot be part of a dimension")
    dup = sorted({c for c in used if used.count(c) > 1})
    if dup:
        raise ValueError(f"columns {dup} appear in more than one dimension")

    for c in used:  # missing text becomes an explicit member instead of a broken relationship
        if _is_text(work[c]):
            work[c] = work[c].fillna("(Unknown)")
    fact_df, tables, rels = work.copy(), {}, []
    for name, cols in dims.items():
        key = f"{name}Key"
        if key in work.columns:
            raise ValueError(f"dimension key '{key}' clashes with an existing column; rename the dimension")
        dim = work[cols].drop_duplicates().sort_values(cols).reset_index(drop=True)
        dim.insert(0, key, np.arange(1, len(dim) + 1))
        fact_df = fact_df.merge(dim, on=cols, how="left").drop(columns=cols)
        tables[name] = dim
        rels.append((fact, key, name, key))
        if len(cols) > 1 and work[cols[0]].nunique() < len(dim):
            notes.append(f"dimension {name}: {len(dim):,} rows but only {work[cols[0]].nunique():,} distinct '{cols[0]}' values, so some '{cols[0]}' values appear with different "
                         f"{', '.join(cols[1:])}; {name}Key identifies the whole combination (check that this is intended)")
    keys = [f"{n}Key" for n in dims]
    fact_df = fact_df[[date] + keys + [c for c in fact_df.columns if c != date and c not in keys]]
    tables = {fact: fact_df, **tables}
    tables["Date"] = make_date_table(f"{work[date].dt.year.min()}-01-01", f"{work[date].dt.year.max()}-12-31", fiscal_start_month)
    rels.append((fact, date, "Date", "Date"))

    files, queries = [], []
    for name, tdf in tables.items():
        p = outdir / f"{name}.csv"
        tdf.to_csv(p, index=False, date_format="%Y-%m-%d")
        files.append(str(p))
        queries.append(m_load_csv(name, p, tdf))
    (outdir / "load_queries.m").write_text(
        "// One query per table. In Power BI Desktop: Home > Transform data > New Source > Blank Query > Advanced Editor,\n"
        "// paste ONE query below, rename the query to the name in its first comment line. Repeat for each table.\n\n"
        + "\n\n".join(queries) + "\n", encoding="utf-8")
    rel_lines = ["# Relationships (many-to-one, single direction)", "", "| From (many) | To (one) |", "|---|---|"]
    rel_lines += [f"| {a}[{b}] | {c}[{d}] |" for a, b, c, d in rels]
    rel_lines += ["", "Mark the `Date` table as a date table (Table tools > Mark as date table > Date column)."]
    (outdir / "relationships.md").write_text("\n".join(rel_lines) + "\n", encoding="utf-8")
    rows = []
    for name, tdf in tables.items():
        for c in tdf.columns:
            role = "key" if c.endswith("Key") or (name == "Date" and c == "Date") else "date" if pd.api.types.is_datetime64_any_dtype(tdf[c]) else (
                "measure column" if name == fact and pd.api.types.is_numeric_dtype(tdf[c]) else "attribute")
            rows.append({"table": name, "column": c, "type": _m_type(tdf[c]).replace("type ", ""), "role": role})
    pd.DataFrame(rows).to_csv(outdir / "data_dictionary.csv", index=False)
    files += [str(outdir / n) for n in ("load_queries.m", "relationships.md", "data_dictionary.csv")]

    checks = {**({"date_agrees_with_year_month_day_columns": not bad_parts} if part_shares else {}), "fact_rows_match": len(fact_df) == len(df), "dates_covered_by_date_table": bool(set(fact_df[date]) <= set(tables["Date"]["Date"])),
              "keys_unique": all(tables[n][f"{n}Key"].is_unique for n in dims), "no_missing_keys": bool(fact_df[keys].notna().all().all()) if keys else True}
    if verify:  # join every dimension back and compare with the input: nothing may be lost or changed
        rebuilt = fact_df.copy()
        for name in dims:
            rebuilt = rebuilt.merge(tables[name], on=f"{name}Key", how="left")
        try:
            for c in used + [c for c in work.columns if c not in used]:
                pd.testing.assert_series_equal(rebuilt[c].reset_index(drop=True), work[c].reset_index(drop=True), check_dtype=False, check_names=False)
            checks["lossless_roundtrip"] = True
        except AssertionError:
            checks["lossless_roundtrip"] = False
    print(f"Power BI star schema written to {outdir}: " + ", ".join(f"{n} ({len(t):,} rows)" for n, t in tables.items()))
    print("  Relationships: " + "; ".join(f"{a}[{b}] -> {c}[{d}]" for a, b, c, d in rels))
    print("  Checks: " + ", ".join(f"{k}={'OK' if v else 'FAILED'}" for k, v in checks.items()))
    for n in notes:
        print("  Note: " + n)
    return Result(tables=tables, files=files, relationships=rels, checks=checks, notes=notes, outdir=str(outdir))


# ------------------------------------------------------------------------ DAX
def _q(table):
    return "'" + str(table).replace("'", "''") + "'"


def _c(table, col):
    return f"{_q(table)}[{str(col).replace(']', ']]')}]"


def _m(name):
    return "[" + str(name).replace("]", "]]") + "]"


_QUALIFIED = re.compile(r"'((?:[^']|'')+)'\[((?:[^\]]|\]\])+)\]")
_BARE = re.compile(r"\[((?:[^\]]|\]\])+)\]")


def check_dax(measures, tables=None):
    """Structural checks for {name: expression}: balanced brackets/quotes, every [Measure] reference is defined,
    and (when tables={'Sales': [columns], ...} is given) every 'Table'[Column] exists. It cannot run DAX."""
    problems = []
    for name, expr in measures.items():
        body = re.sub(r'"[^"]*"', '""', expr)
        if body.count("(") != body.count(")"):
            problems.append(f"[{name}]: unbalanced parentheses")
        if re.sub(r"''", "", body).count("'") % 2:
            problems.append(f"[{name}]: unbalanced single quotes in a table name")
        if re.search(r"\{[A-Za-z_]+\}", expr):
            problems.append(f"[{name}]: unresolved placeholder")
        for t, col in _QUALIFIED.findall(body):
            t, col = t.replace("''", "'"), col.replace("]]", "]")
            if tables is not None and (t not in tables or col not in tables[t]):
                problems.append(f"[{name}]: unknown column {t}[{col}]")
        rest = _QUALIFIED.sub(" ", body)
        for ref in _BARE.findall(rest):
            if ref.replace("]]", "]") not in measures:
                problems.append(f"[{name}]: refers to undefined measure [{ref}]")
    return problems


def dax_measures(fact="Sales", value="Revenue", date_table="Date", date_col="Date", label=None, which=None,
                 rank_by=None, fiscal_year_end=None, month_col="YearMonth", path=None, tables=None):
    """Tested DAX templates for one numeric column. which = any of: total rows avg ly yoy ytd mtd mom running moving pct
    (default all). rank_by=('Product', 'Product') adds a rank measure. fiscal_year_end='3/31' for fiscal years.
    Needs a gap-free Date table marked as a date table (export_star_schema creates one).
    Result keys: measures (name -> expression), text, problems, path."""
    L = label or value
    T, D = f"Total {L}", _c(date_table, date_col)
    base = {
        "total": {T: f"SUM({_c(fact, value)})"},
        "rows": {"Row Count": f"COUNTROWS({_q(fact)})"},
        "avg": {"Row Count": f"COUNTROWS({_q(fact)})", f"Average {L} per Row": f"DIVIDE({_m(T)}, {_m('Row Count')})"},
        "ly": {f"{L} LY": f"CALCULATE({_m(T)}, SAMEPERIODLASTYEAR({D}))"},
        "yoy": {f"{L} LY": f"CALCULATE({_m(T)}, SAMEPERIODLASTYEAR({D}))", f"{L} YoY %": f"DIVIDE({_m(T)} - {_m(L + ' LY')}, {_m(L + ' LY')})"},
        "ytd": {f"{L} YTD": f"TOTALYTD({_m(T)}, {D}" + (f', "{fiscal_year_end}"' if fiscal_year_end else "") + ")"},
        "mtd": {f"{L} MTD": f"TOTALMTD({_m(T)}, {D})"},
        "mom": {f"{L} Previous Month": f"CALCULATE({_m(T)}, DATEADD({D}, -1, MONTH))",
                f"{L} MoM %": f"DIVIDE({_m(T)} - {_m(L + ' Previous Month')}, {_m(L + ' Previous Month')})"},
        "running": {f"{L} Running Total": f"CALCULATE({_m(T)}, FILTER(ALL({D}), {D} <= MAX({D})))"},
        "moving": {f"{L} 3M Moving Avg": f"VAR _window = DATESINPERIOD({D}, MAX({D}), -3, MONTH)\nRETURN\n    DIVIDE(CALCULATE({_m(T)}, _window), CALCULATE(DISTINCTCOUNT({_c(date_table, month_col)}), _window))"},
        "pct": {f"{L} % of Total": f"DIVIDE({_m(T)}, CALCULATE({_m(T)}, ALLSELECTED()))"},
    }
    chosen = list(which) if which else list(base)
    unknown = [w for w in chosen if w not in base]
    if unknown:
        raise ValueError(f"unknown measure group(s) {unknown}; choose from {list(base)}")
    measures = dict(base["total"])
    for w in chosen:
        measures.update(base[w])
    if rank_by:
        rt, rc = rank_by
        measures[f"{L} Rank by {rc}"] = f"RANKX(ALL({_c(rt, rc)}), {_m(T)})"
    if tables is None:
        tables = {fact: [value], date_table: [date_col, month_col]}
        if rank_by:
            tables.setdefault(rank_by[0], []).append(rank_by[1])
    problems = check_dax(measures, tables)
    lines = [f"// DAX measures for {fact}[{value}]. In Power BI Desktop: Report view > select the '{fact}' table > New measure, paste one measure at a time.",
             "// Needs a Date table marked as a date table. Percent measures: set Format to Percentage.", ""]
    for name, expr in measures.items():
        pct = "%" in name
        lines.append(f"// {'format: 0.0%' if pct else 'format: #,0.00'}")
        lines.append(f"{name} = " + (expr if "\n" not in expr else "\n    " + expr.replace("\n", "\n    ")))
        lines.append("")
    text = "\n".join(lines)
    if path:
        Path(path).write_text(text, encoding="utf-8")
    print(f"DAX: {len(measures)} measures for {fact}[{value}]" + (f" written to {path}" if path else "") + (f"; PROBLEMS: {problems}" if problems else "; structure checks passed"))
    return Result(measures=measures, text=text, problems=problems, path=str(path) if path else None)


# ----------------------------------------------------------- expected values
def expected_values(df, date, value, by=None, fiscal_start_month=1, label=None, path=None):
    """What your Power BI visuals should show, computed independently in pandas with the same definitions as the DAX
    templates: totals by Year (with last year and YoY %) and by Month (YTD, MTD, previous month, MoM %, running total,
    3-month moving average, % of total), plus totals/% of total/rank by the column `by`.
    Compare these numbers with a table visual in Power BI. Result keys: year, month, by (DataFrames), path."""
    L = label or value
    date, value = _col(df, date, "date"), _col(df, value, "value")
    d = df[[date, value] + ([_col(df, by, "by")] if by else [])].dropna(subset=[date, value]).copy()
    d[date] = pd.to_datetime(d[date])
    fm = int(fiscal_start_month)
    first_year = int(d[date].dt.year.min())
    monthly = d.groupby(d[date].dt.to_period("M"))[value].sum()
    rng = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    monthly = monthly.reindex(rng)
    table_start = pd.Period(f"{first_year}-01", "M")

    yearly = d.groupby(d[date].dt.year)[value].sum().rename(f"Total {L}").to_frame()
    yearly[f"{L} LY"] = yearly[f"Total {L}"].shift(1).where(pd.Series(yearly.index - 1, index=yearly.index).isin(yearly.index))
    yearly[f"{L} YoY %"] = (yearly[f"Total {L}"] - yearly[f"{L} LY"]) / yearly[f"{L} LY"].replace(0, np.nan)
    yearly.index.name = "Year"

    m = pd.DataFrame({"YearMonth": [str(p) for p in rng], f"Total {L}": monthly.to_numpy()}, index=rng)
    ly = pd.Series([monthly.get(p - 12, np.nan) for p in rng], index=rng)
    prev = pd.Series([monthly.get(p - 1, np.nan) for p in rng], index=rng)
    m[f"{L} LY"], m[f"{L} YoY %"] = ly, (m[f"Total {L}"] - ly) / ly.replace(0, np.nan)
    fy = pd.Series([p.year + (1 if (fm > 1 and p.month >= fm) else 0) for p in rng], index=rng)
    m[f"{L} YTD"] = m[f"Total {L}"].fillna(0).groupby(fy).cumsum()
    m[f"{L} MTD"] = m[f"Total {L}"]
    m[f"{L} Previous Month"], m[f"{L} MoM %"] = prev, (m[f"Total {L}"] - prev) / prev.replace(0, np.nan)
    m[f"{L} Running Total"] = m[f"Total {L}"].fillna(0).cumsum()
    window = pd.Series([sum(0.0 if pd.isna(monthly.get(p - k)) else monthly.get(p - k) for k in range(3)) for p in rng], index=rng)  # a month without sales counts as 0
    months_in_table = pd.Series([sum(1 for k in range(3) if (p - k) >= table_start) for p in rng], index=rng)
    m[f"{L} 3M Moving Avg"] = window / months_in_table
    m[f"{L} % of Total"] = m[f"Total {L}"] / m[f"Total {L}"].sum()
    m = m.reset_index(drop=True)

    out = {"year": yearly.reset_index(), "month": m}
    if by:
        g = d.groupby(by)[value].sum().rename(f"Total {L}").to_frame()
        g[f"{L} % of Total"] = g[f"Total {L}"] / g[f"Total {L}"].sum()
        g["Rank"] = g[f"Total {L}"].rank(method="min", ascending=False).astype(int)
        out["by"] = g.sort_values("Rank").reset_index()
    if path:
        from analyst_tools import save_excel

        pct = {f"{L} YoY %": "0.0%", f"{L} MoM %": "0.0%", f"{L} % of Total": "0.0%", "Year": "0", "Rank": "0"}
        save_excel(path, {"By Year": out["year"], "By Month": out["month"], **({f"By {by}"[:31]: out["by"]} if by else {})}, formats=pct)
    print(f"Expected values for {L}: " + "; ".join(f"{int(r.Year)}: {r[f'Total {L}']:,.2f}" for _, r in out["year"].iterrows()))
    return Result(year=out["year"], month=out["month"], by=out.get("by"), path=str(path) if path else None)


# --------------------------------------------------------------- full package
def build_powerbi_package(df, outdir, fact="Sales", date="Date", value="Revenue", dims=None, fiscal_start_month=1, rank_by=None):
    """One call: star-schema CSVs, Power Query (M) load script, DAX measures, relationships guide,
    expected_values.xlsx and a README_PowerBI.md with the steps to follow in Power BI Desktop."""
    outdir = Path(outdir)
    value = _col(df, value, "value")
    if dims and value in [c for v in dims.values() for c in v]:
        raise ValueError(f"'{value}' is the measure column and cannot also be in a dimension")
    star = export_star_schema(df, outdir, fact, date, dims, fiscal_start_month)
    fye = None
    if fiscal_start_month > 1:
        em = fiscal_start_month - 1
        fye = f"{em}/{calendar.monthrange(2001, em)[1]}"
    dx = dax_measures(fact, value, "Date", "Date", rank_by=rank_by, fiscal_year_end=fye, path=outdir / "measures.dax")
    ev = expected_values(df, date, value, by=rank_by[1] if rank_by and rank_by[1] in df.columns else None,
                         fiscal_start_month=fiscal_start_month, path=outdir / "expected_values.xlsx")
    readme = f"""# Power BI package for {fact}

Files in this folder: one CSV per table, `load_queries.m`, `relationships.md`, `measures.dax`,
`expected_values.xlsx`, `data_dictionary.csv`.

## Steps in Power BI Desktop
1. **Load the tables.** Home > Transform data > New Source > Blank Query > Advanced Editor. Paste ONE query from
   `load_queries.m` (the code under one `// Query name:` line), rename the query to that name, and repeat for every
   table. (Quick alternative: Get data > Text/CSV for each file; the .m files also set the column types.)
   The queries contain the full path to this folder; edit the path if you move it. Then Close & Apply.
2. **Relationships.** Model view. Power BI usually detects them from the key names; otherwise create them as listed in
   `relationships.md` (many-to-one, single direction). Mark `Date` as a date table (Table tools > Mark as date table).
3. **Measures.** Select the `{fact}` table, New measure, paste one measure at a time from `measures.dax`.
   Set percent measures to Percentage format. Hide the raw `{value}` column if you like.
4. **Check the numbers.** Add a table visual with `Date[Year]`, `Date[YearMonth]` and the measures. Compare with
   `expected_values.xlsx` (computed separately in pandas). Differences mean a wrong relationship, filter or measure.

## What was and was not verified
The CSVs were checked (row counts, unique keys, lossless split, dates covered). The M and DAX text was checked for
structure only. It could not be run outside Power BI, so step 4 is how you confirm it.
"""
    (outdir / "README_PowerBI.md").write_text(readme, encoding="utf-8")
    files = star.files + [str(outdir / "measures.dax"), str(outdir / "expected_values.xlsx"), str(outdir / "README_PowerBI.md")]
    print(f"Power BI package ready in {outdir}. Start with README_PowerBI.md")
    return Result(star=star, dax=dx, expected=ev, files=files, outdir=str(outdir))
