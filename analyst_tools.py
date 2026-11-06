"""
analyst_tools.py - tested helper functions for the local data-analyst agent.

Small local models make fewer mistakes when they can call one reliable function
instead of writing common analyst code from scratch. You can also use these
yourself in any notebook:

    from analyst_tools import *
    df = clean_columns(pd.read_csv("data.csv"), dayfirst=True)
    profile(df)
    compare_groups(df, "revenue", "region")
    regress(df, "revenue", ["units", "region"])
    top_correlations(df)
    quick_chart(df, x="month", y="revenue", hue="region", kind="bar")
    save_excel("out.xlsx", {"Data": df})

Functions print a short readable summary AND return a Result (a dict) so the
numbers can be reused in code.
"""
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "Result",
    "clean_columns",
    "profile",
    "compare_groups",
    "regress",
    "top_correlations",
    "crosstab_test",
    "compare_proportions",
    "trend_forecast",
    "quick_chart",
    "save_excel",
    "inspect_workbook",
    "read_excel_smart",
]

_NA_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "--"}


class Result(dict):
    """A dict with attribute access and a short repr (keeps the agent's context small)."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"Result has no '{name}'. Available: {list(self.keys())}") from None

    def __repr__(self):
        return f"Result(keys={list(self.keys())})"


def _col(df, x, arg="column"):
    """Accept a column NAME (or a Series taken from df, using its name); explain clearly otherwise."""
    if isinstance(x, pd.Series) and x.name in df.columns:
        x = x.name
    if isinstance(x, (pd.Series, pd.DataFrame, list, tuple, set, dict, np.ndarray)):
        raise TypeError(f"{arg} must be a column NAME such as {list(df.columns)[:3]}, not {type(x).__name__}. Pass the name, not the data.")
    if x not in df.columns:
        raise KeyError(f"column '{x}' not found (argument {arg}). Available columns: {list(df.columns)}")
    return x


def _num(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    x = float(x)
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.4g}"


def _p(p):
    return "<0.0001" if p < 1e-4 else f"={p:.4f}"


# ------------------------------------------------------------------ cleaning
def _date_order(s, dayfirst):
    """Day-first or month-first for d/m/y text dates, decided from the values. A first part above 12 can only be a
    day; a second part above 12 can only be a day (so the first part is the month). Returns (dayfirst, how, warning)."""
    parts = s.dropna().astype(str).str.extract(r"^\s*(\d{1,2})[/\-.](\d{1,2})[/\-.]\d{2,4}")
    a, b = pd.to_numeric(parts[0], errors="coerce"), pd.to_numeric(parts[1], errors="coerce")
    matched, n_day, n_month = int(a.notna().sum()), int((a > 12).sum()), int((b > 12).sum())
    if matched == 0:  # ISO dates, month names, ... : no ambiguity to resolve
        return bool(dayfirst), None, None
    if dayfirst is None:
        if n_day and not n_month:
            return True, f"day-first, detected: {n_day:,} values have a first part above 12", None
        if n_month and not n_day:
            return False, f"month-first, detected: {n_month:,} values have a second part above 12", None
        if n_day and n_month:
            return n_day >= n_month, "mixed formats", f"{n_day:,} values look day-first and {n_month:,} look month-first; ambiguous ones follow the majority. Check the source"
        return False, "month-first assumed", ("no value has a part above 12, so day-first vs month-first cannot be told from the data; assumed month-first. "
                                              "Pass dayfirst=True if the dates are DD/MM/YYYY")
    how, warn = ("day-first" if dayfirst else "month-first"), None
    if dayfirst and n_month:
        warn = f"dayfirst=True was given but {n_month:,} values have a second part above 12 (cannot be a month): the dates look month-first"
    elif not dayfirst and n_day:
        warn = f"dayfirst=False was given but {n_day:,} values have a first part above 12 (cannot be a month): the dates look day-first"
    return bool(dayfirst), how, warn


def clean_columns(df, dayfirst=None, rename=False, verbose=True):
    """Strip text, turn numbers/dates stored as text into real numbers/dates.
    Returns a cleaned COPY and prints what changed. Column names are KEPT as they are
    unless rename=True (which makes them snake_case: 'Revenue ($)' -> 'revenue').
    dayfirst=None (default) detects day-first vs month-first from the values and says which it chose; pass True for
    DD/MM/YYYY or False for MM/DD/YYYY only when you know. A WARNING is printed if the data contradicts the choice."""
    out = df.copy()
    log = []
    if rename:
        seen, names = {}, []
        for c in out.columns:
            n = re.sub(r"[^0-9a-zA-Z]+", "_", str(c).strip()).strip("_").lower() or "col"
            if n in seen:
                seen[n] += 1
                n = f"{n}_{seen[n]}"
            else:
                seen[n] = 0
            names.append(n)
        renamed = {o: n for o, n in zip(out.columns, names) if str(o) != n}
        if renamed:
            log.append("renamed columns: " + ", ".join(f"{o!r}->{n}" for o, n in renamed.items()))
        out.columns = names

    for c in out.columns:
        s = out[c]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue
        s = s.astype("string").str.strip()
        s = s.mask(s.str.lower().isin(_NA_TOKENS))
        non_null = int(s.notna().sum())
        if non_null == 0:
            out[c] = s.astype(object).where(s.notna(), np.nan)
            continue

        # numbers stored as text: "1,234", "$12.50", "15%"
        t = s.str.replace(r"[,$€£₹\s]", "", regex=True)
        pct_share = float(t.str.endswith("%").fillna(False).sum()) / non_null
        num = pd.to_numeric(t.str.rstrip("%"), errors="coerce")
        if num.notna().sum() >= 0.9 * non_null:
            if pct_share >= 0.5:
                num = num / 100
            bad = non_null - int(num.notna().sum())
            out[c] = num.astype(float)
            msg = f"{c}: text -> number"
            if pct_share >= 0.5:
                msg += " (percent -> fraction)"
            if bad:
                msg += f" ({bad} unparseable values became missing)"
            log.append(msg)
            continue

        # dates stored as text
        has_digit = float(s.str.contains(r"\d", regex=True).fillna(False).sum()) / non_null
        if has_digit >= 0.9:
            raw = s.astype(object).where(s.notna(), None)
            dayfirst_used, how, date_warn = _date_order(s, dayfirst)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    dt = pd.to_datetime(raw, errors="coerce", dayfirst=dayfirst_used, format="mixed")
                except (TypeError, ValueError):
                    dt = pd.to_datetime(raw, errors="coerce", dayfirst=dayfirst_used)
            if dt.notna().sum() >= 0.9 * non_null:
                bad = non_null - int(dt.notna().sum())
                out[c] = dt
                log.append(f"{c}: text -> date" + (f" ({how})" if how else "") + (f" ({bad} unparseable became missing)" if bad else ""))
                if date_warn:
                    log.append(f"WARNING {c}: {date_warn}")
                continue

        out[c] = s.astype(object).where(s.notna(), np.nan)

    dups = int(out.duplicated().sum())
    if dups:
        log.append(f"{dups} exact duplicate rows found (this function does not remove them; use df.drop_duplicates())")
    if verbose:
        print("clean_columns: " + ("no changes needed" if not log else "changes made (column names kept)" if not rename else "changes made"))
        for line in log:
            print("  - " + line)
    return out


def profile(df, name="data", max_cat=5):
    """Print shape, column types, missing values, basic stats and common data problems."""
    print(f"{name}: {len(df):,} rows x {df.shape[1]} columns; {int(df.duplicated().sum()):,} duplicate rows")
    hints = []
    for c in df.columns:
        s = df[c]
        miss = s.isna().mean() * 100
        line = f"- {c} [{s.dtype}] missing {miss:.1f}%, unique {s.nunique():,}"
        if pd.api.types.is_bool_dtype(s):
            line += ", values " + ", ".join(f"{k} ({v})" for k, v in s.value_counts().head(max_cat).items())
        elif pd.api.types.is_numeric_dtype(s):
            n_inf = int(np.isinf(s.to_numpy(float, na_value=np.nan)).sum()) if s.notna().any() else 0
            if n_inf:
                hints.append(f"'{c}' contains {n_inf} infinite values (division by zero?) - tests on it will fail")
            if s.notna().any():
                line += (
                    f", min {_num(s.min())}, median {_num(s.median())}, mean {_num(s.mean())}, "
                    f"max {_num(s.max())}, sd {_num(s.std())}"
                )
        elif pd.api.types.is_datetime64_any_dtype(s):
            if s.notna().any():
                line += f", range {s.min():%Y-%m-%d} to {s.max():%Y-%m-%d}"
        else:
            line += ", top: " + ", ".join(f"{k} ({v})" for k, v in s.value_counts().head(max_cat).items())
            txt = s.dropna().astype(str).str.replace(r"[,$%\s]", "", regex=True)
            if len(txt) and pd.to_numeric(txt, errors="coerce").notna().mean() > 0.9:
                hints.append(f"'{c}' looks numeric but is stored as text -> use clean_columns(df)")
        print(line)
        if s.nunique(dropna=True) <= 1 and len(s):
            hints.append(f"'{c}' is constant or empty")
        if miss > 30:
            hints.append(f"'{c}' is {miss:.0f}% missing")
    if hints:
        print("Possible problems:")
        for h in hints:
            print("  ! " + h)


# ----------------------------------------------------------------- group tests
def _label(value, cuts, names):
    for cut, name in zip(cuts, names):
        if abs(value) < cut:
            return name
    return names[-1]


def _welch_anova(vals):
    """Welch's ANOVA (does not assume equal variances). Returns (F, df1, df2, p)."""
    from scipy import stats

    k = len(vals)
    n = np.array([len(v) for v in vals], float)
    m = np.array([v.mean() for v in vals])
    s2 = np.array([v.var(ddof=1) for v in vals])
    w = n / s2
    sw = w.sum()
    mw = (w * m).sum() / sw
    lam = ((1 - w / sw) ** 2 / (n - 1)).sum()
    f = ((w * (m - mw) ** 2).sum() / (k - 1)) / (1 + 2 * (k - 2) / (k**2 - 1) * lam)
    df2 = (k**2 - 1) / (3 * lam)
    return float(f), k - 1, float(df2), float(stats.f.sf(f, k - 1, df2))


def _holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, float)
    m = len(p)
    adj, running = np.empty(m), 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def compare_groups(df, value, group, groups=None, test="auto", alpha=0.05):
    """Compare a numeric column across the groups of another column.
    test: 'auto' (checks assumptions and picks), 'welch', 'mannwhitney', 'anova', 'kruskal'.
    Prints group stats, assumption checks, the test, an effect size and (for 3+ groups) post-hoc pairs.
    Result keys: test, statistic, p, effect_size, effect_name, significant, plus t/d/diff/ci for 2-group Welch."""
    from scipy import stats

    value, group = _col(df, value, "value"), _col(df, group, "group")
    d = df[[value, group]].dropna()
    d = d.astype({value: float})
    bad = ~np.isfinite(d[value])
    if bad.any():
        print(f"  WARNING: ignored {int(bad.sum())} infinite values in '{value}' (check how they arose, e.g. division by zero)")
        d = d[~bad]
    available = sorted(d[group].unique(), key=str)
    if groups is not None:
        groups = list(groups)
        if any(isinstance(g, (pd.Series, pd.DataFrame, np.ndarray, list, tuple, set)) for g in groups):
            raise ValueError(f"groups must be a list of group NAMES from the '{group}' column, e.g. {available[:4]}, not data. Leave it out to use all groups.")
        missing = [g for g in groups if g not in available]
        if missing:
            raise ValueError(f"groups not found in '{group}': {missing}. Available: {available}")
    if groups is None:
        groups = available
        if len(groups) > 10:
            raise ValueError(f"{len(groups)} groups found; pass groups=[...] to choose which ones to compare")
    samples = {g: d.loc[d[group] == g, value].to_numpy() for g in groups}
    small = [g for g, v in samples.items() if len(v) < 2]
    if len(samples) < 2 or small:
        raise ValueError(f"need at least 2 groups with 2+ values each; problem groups: {small}")
    names, vals, k = list(samples), list(samples.values()), len(samples)
    desc = pd.DataFrame(
        [dict(group=g, n=len(v), mean=v.mean(), sd=v.std(ddof=1), median=float(np.median(v))) for g, v in samples.items()]
    )

    normal_ps = [stats.shapiro(v).pvalue for v in vals if 3 <= len(v) <= 5000 and np.ptp(v) > 0]
    normal = all(p > alpha for p in normal_ps)
    min_n = min(len(v) for v in vals)
    lev_p = float(stats.levene(*vals, center="median").pvalue) if all(np.ptp(v) > 0 for v in vals) else float("nan")

    if test == "auto":
        parametric = normal or min_n >= 30
        test = ("welch" if k == 2 else "anova") if parametric else ("mannwhitney" if k == 2 else "kruskal")
    if k > 2 and test == "welch":
        test = "anova"
    if k == 2 and test == "anova":
        test = "welch"
    if k > 2 and test == "mannwhitney":
        test = "kruskal"
    if k == 2 and test == "kruskal":
        test = "mannwhitney"

    res = Result(test=test, groups=desc, alpha=alpha, normality_ok=normal, levene_p=lev_p)
    print(f"{value} by {group}: {' vs '.join(map(str, names))}")
    for _, r in desc.iterrows():
        print(f"  {r['group']}: n={int(r['n'])} mean={_num(r['mean'])} sd={_num(r['sd'])} median={_num(r['median'])}")
    print(
        f"  Assumptions: normality {'OK' if normal else 'violated'}"
        + (f" (smallest Shapiro p{_p(min(normal_ps))})" if normal_ps else "")
        + f", equal variances {'n/a' if np.isnan(lev_p) else ('OK' if lev_p > alpha else 'violated')}"
        + (f" (Levene p{_p(lev_p)})" if not np.isnan(lev_p) else "")
    )

    if test == "welch":
        a, b = vals
        t, p = stats.ttest_ind(a, b, equal_var=False)
        v1, v2 = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        if v1 + v2 == 0:
            raise ValueError("no variation in either group")
        diff, se = a.mean() - b.mean(), np.sqrt(v1 + v2)
        dfw = (v1 + v2) ** 2 / (v1**2 / (len(a) - 1) + v2**2 / (len(b) - 1))
        tc = stats.t.ppf(1 - alpha / 2, dfw)
        ci = (float(diff - tc * se), float(diff + tc * se))
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        eff, eff_name = float(diff / sp), "Cohen's d"
        stat = float(t)
        label = _label(eff, (0.2, 0.5, 0.8), ("negligible", "small", "medium", "large"))
        res.update(t=stat, d=eff, diff=float(diff), ci=ci, welch_df=float(dfw))
        test_line = f"Welch t-test: t={stat:.4f}, df={dfw:.1f}, p{_p(p)}"
        extra = f"  Mean difference ({names[0]} - {names[1]}) = {_num(diff)}, {int((1 - alpha) * 100)}% CI [{_num(ci[0])}, {_num(ci[1])}]"
    elif test == "mannwhitney":
        a, b = vals
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        stat, p = float(u.statistic), float(u.pvalue)
        eff, eff_name = 2 * stat / (len(a) * len(b)) - 1, "rank-biserial r"
        label = _label(eff, (0.1, 0.3, 0.5), ("negligible", "small", "medium", "large"))
        test_line = f"Mann-Whitney U test: U={stat:.1f}, p{_p(p)}"
        extra = ""
    elif test == "anova":
        if all(v.var(ddof=1) > 0 for v in vals):
            stat, _, dfd, p = _welch_anova(vals)
            test_name = f"Welch ANOVA (df2={dfd:.1f})"
        else:
            stat, p = (float(x) for x in stats.f_oneway(*vals))
            test_name = "one-way ANOVA"
        allv = np.concatenate(vals)
        ssb = sum(len(v) * (v.mean() - allv.mean()) ** 2 for v in vals)
        eff, eff_name = float(ssb / ((allv - allv.mean()) ** 2).sum()), "eta squared"
        label = _label(eff, (0.01, 0.06, 0.14), ("negligible", "small", "medium", "large"))
        test_line = f"{test_name}: F={stat:.4f}, p{_p(p)}"
        extra = ""
    elif test == "kruskal":
        h, p = stats.kruskal(*vals)
        stat, p = float(h), float(p)
        n_all = sum(len(v) for v in vals)
        eff, eff_name = float(stat * (n_all + 1) / (n_all**2 - 1)), "epsilon squared"
        label = _label(eff, (0.01, 0.06, 0.14), ("negligible", "small", "medium", "large"))
        test_line = f"Kruskal-Wallis: H={stat:.4f}, p{_p(p)}"
        extra = ""
    else:
        raise ValueError("test must be auto, welch, mannwhitney, anova or kruskal")

    if not (np.isfinite(stat) and np.isfinite(p)):
        raise ValueError("the test returned NaN/inf: check for infinite values, constant groups or too few rows per group")
    sig = p < alpha
    res.update(statistic=stat, p=float(p), effect_size=float(eff), effect_name=eff_name, significant=bool(sig))
    print(f"  Test: {test_line} -> {'significant' if sig else 'not significant'} at {alpha}")
    print(f"  Effect size: {eff_name} = {eff:.3f} ({label})")
    if extra:
        print(extra)

    if k > 2 and sig:
        from itertools import combinations

        pairs, raw = [], []
        for i, j in combinations(range(k), 2):
            if test == "anova":
                pv = stats.ttest_ind(vals[i], vals[j], equal_var=False).pvalue
            else:
                pv = stats.mannwhitneyu(vals[i], vals[j], alternative="two-sided").pvalue
            pairs.append((names[i], names[j]))
            raw.append(pv)
        adj = _holm(raw)
        post = pd.DataFrame({"group_1": [a for a, _ in pairs], "group_2": [b for _, b in pairs], "p_adjusted": adj})
        res["posthoc"] = post
        sig_pairs = post[post["p_adjusted"] < alpha].sort_values("p_adjusted").head(10)
        print("  Post-hoc (Holm-adjusted), significant pairs: " + (", ".join(f"{r.group_1} vs {r.group_2} (p{_p(r.p_adjusted)})" for r in sig_pairs.itertuples()) or "none"))
    return res


# ------------------------------------------------------------------ regression
_UNSET = object()


class _Params(pd.Series):
    """Coefficients by name. A short name finds the full one ('West' -> 'Region[West]'),
    and an unknown name raises an error listing the real names instead of returning None."""

    @property
    def _constructor(self):
        return pd.Series

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            if isinstance(key, str):
                hits = [n for n in self.index if str(n).endswith(f"[{key}]") or str(n).endswith(f"_{key}")]
                if len(hits) == 1:
                    return super().__getitem__(hits[0])
            raise KeyError(f"'{key}' is not a coefficient name. Available: {list(self.index)}") from None

    def get(self, key, default=_UNSET):
        try:
            return self[key]
        except KeyError:
            if default is _UNSET:
                raise
            return default


def _design(d, X):
    """Design matrix with intercept; text columns become dummies (first level alphabetically = reference)."""
    cols, names, refs, levels_by_col = [np.ones(len(d))], ["Intercept"], [], {}
    for c in X:
        s = d[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):  # True/False dummies are 0/1 numbers
            cols.append(s.to_numpy(float))
            names.append(c)
        else:
            sv = s.astype(str)
            levels = sorted(sv.unique())
            refs.append(f"{c}={levels[0]}")
            levels_by_col[c] = levels[1:]
            for lvl in levels[1:]:
                cols.append((sv == lvl).to_numpy(float))
                names.append(f"{c}[{lvl}]")
    return np.column_stack(cols), names, refs, levels_by_col


def regress(df, y, X, robust=False, alpha=0.05):
    """OLS regression of numeric y on predictors X (a column name or a list).
    Text columns become categories named like Region[West] (reference = first level alphabetically);
    True/False or 0/1 columns are used as they are.
    robust=True uses heteroskedasticity-robust (HC3) standard errors.
    Prints R-squared, coefficients with CIs and diagnostics.
    Result keys: predict(new_df) -> predictions for new rows, params (coefficient by name, e.g. .params['Units']), coef (DataFrame), r2, adj_r2, n, f_p, bp_p, resid_normal_p, max_vif, resid, fitted."""
    from scipy import stats

    X = [X] if isinstance(X, (str, pd.Series)) else list(X)
    y, X = _col(df, y, "y"), [_col(df, c, "X") for c in X]
    d = df[[y] + X].dropna().copy()
    Xm, names, refs, levels_by_col = _design(d, X)
    yv = d[y].to_numpy(float)
    if not (np.isfinite(Xm).all() and np.isfinite(yv).all()):
        raise ValueError("the data contains infinite values; remove or fix them before regressing")
    n, p = Xm.shape
    rank = int(np.linalg.matrix_rank(Xm))
    if n < p + 5:
        raise ValueError("not enough complete rows for this many predictors")
    dof = n - rank
    xtx_inv = np.linalg.pinv(Xm.T @ Xm)
    beta = xtx_inv @ Xm.T @ yv
    fitted = Xm @ beta
    resid = yv - fitted
    ss_res, ss_tot = float(resid @ resid), float(((yv - yv.mean()) ** 2).sum())
    if robust:
        h = np.einsum("ij,jk,ik->i", Xm, xtx_inv, Xm)
        w = resid**2 / (1 - h) ** 2
        cov = xtx_inv @ (Xm.T * w) @ Xm @ xtx_inv
        crit = stats.norm.ppf(1 - alpha / 2)
    else:
        cov = ss_res / dof * xtx_inv
        crit = stats.t.ppf(1 - alpha / 2, dof)
    se = np.sqrt(np.diag(cov))
    tstat = beta / se
    pvals = 2 * (stats.norm.sf(abs(tstat)) if robust else stats.t.sf(abs(tstat), dof))
    coef = pd.DataFrame(
        {"coef": beta, "std_err": se, "p": pvals, "ci_low": beta - crit * se, "ci_high": beta + crit * se}, index=names
    )
    r2 = 1 - ss_res / ss_tot
    adj = 1 - (1 - r2) * (n - 1) / dof
    f_p = float(stats.f.sf(((ss_tot - ss_res) / (rank - 1)) / (ss_res / dof), rank - 1, dof)) if rank > 1 else float("nan")

    # diagnostics
    bp_p = resid_p = max_vif = None
    if rank > 1:
        u2 = resid**2
        g = np.linalg.lstsq(Xm, u2, rcond=None)[0]
        r2_aux = 1 - float(((u2 - Xm @ g) ** 2).sum()) / float(((u2 - u2.mean()) ** 2).sum() or np.nan)
        bp_p = float(stats.chi2.sf(n * r2_aux, rank - 1))  # Breusch-Pagan, Koenker form
    if n <= 5000:
        try:
            resid_p = float(stats.shapiro(resid).pvalue)
        except Exception:
            pass
    if p > 2:
        vifs = []
        for j in range(1, p):
            others = np.delete(Xm, j, axis=1)
            gj = np.linalg.lstsq(others, Xm[:, j], rcond=None)[0]
            ssr = float(((Xm[:, j] - others @ gj) ** 2).sum())
            sst = float(((Xm[:, j] - Xm[:, j].mean()) ** 2).sum())
            vifs.append(np.inf if ssr <= 1e-12 * max(sst, 1e-12) else sst / ssr)
        max_vif = float(np.max(vifs))

    print(f"OLS: {y} ~ {' + '.join(X)} (n={n}{', robust HC3 errors' if robust else ''})")
    if refs:
        print("  Reference levels: " + ", ".join(refs))
    print(f"  R2={r2:.4f}, adjusted R2={adj:.4f}, overall F-test p{_p(f_p) if not np.isnan(f_p) else ' n/a'}")
    print(coef.to_string(float_format=lambda v: f"{v:.4g}"))
    notes = []
    if rank < p:
        notes.append(f"predictors are perfectly collinear (rank {rank} < {p} columns); drop one")
    if bp_p is not None and bp_p < alpha and not robust:
        notes.append(f"Breusch-Pagan p{_p(bp_p)}: unequal residual variance -> consider robust=True")
    if resid_p is not None and resid_p < alpha:
        notes.append(f"residuals not normal (Shapiro p{_p(resid_p)}); usually fine for large n, check a residual plot")
    if max_vif is not None and max_vif > 10:
        notes.append(f"max VIF={max_vif:.1f}: predictors strongly collinear, coefficients unreliable")
    print("  Diagnostics: " + ("; ".join(notes) if notes else "no problems flagged"))
    def predict(new):
        """Predicted y for new rows (a DataFrame with the same predictor columns)."""
        new = new if isinstance(new, pd.DataFrame) else pd.DataFrame(new)
        cols = [np.ones(len(new))]
        for c in X:
            if c in levels_by_col:
                cols += [(new[c].astype(str) == lvl).to_numpy(float) for lvl in levels_by_col[c]]
            else:
                cols.append(new[c].to_numpy(float))
        return pd.Series(np.column_stack(cols) @ beta, index=new.index)

    return Result(predict=predict, params=_Params(coef["coef"]), coef=coef, r2=float(r2), adj_r2=float(adj), n=n, f_p=f_p, bp_p=bp_p, resid_normal_p=resid_p, max_vif=max_vif, resid=resid, fitted=fitted)


def top_correlations(df, n=8):
    """Print the n strongest numeric correlations (Pearson r, Spearman rho, p-value, n)."""
    from scipy import stats

    num = df.select_dtypes("number").replace([np.inf, -np.inf], np.nan)
    num = num.loc[:, num.nunique() > 1]
    if num.shape[1] > 40:
        num = num.iloc[:, :40]
    if num.shape[1] < 2:
        raise ValueError("need at least 2 numeric columns with variation")
    pear, spear = num.corr(), num.corr(method="spearman")
    cols = list(num.columns)
    pairs = [(cols[i], cols[j], pear.iloc[i, j], spear.iloc[i, j]) for i in range(len(cols)) for j in range(i + 1, len(cols))]
    pairs.sort(key=lambda t: -(abs(t[2]) if not np.isnan(t[2]) else 0))
    rows = []
    for a, b, r, rho in pairs[:n]:
        sub = num[[a, b]].dropna()
        p = stats.pearsonr(sub[a], sub[b])[1] if len(sub) > 2 else np.nan
        rows.append(dict(var1=a, var2=b, pearson_r=r, spearman_rho=rho, p_value=p, n=len(sub)))
    table = pd.DataFrame(rows)
    print(f"Top {len(table)} correlations (of {len(pairs)} pairs):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return Result(table=table)


# ------------------------------------------------------------ categorical tests
def _binary(series, positive=None):
    """Boolean Series for a yes/no style column (Yes/No, True/False, 1/0, or == positive)."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    if positive is not None:
        return series == positive
    return series.astype(str).str.strip().str.lower().isin({"yes", "y", "true", "t", "1", "1.0"})


def _pfmt(p):
    return f"{p:.4e}" if p < 1e-4 else f"{p:.4f}"


def crosstab_test(df, a, b, alpha=0.05):
    """Chi-square test of independence between two categorical columns (rows are the observations).
    Prints the table, the test, Cramer's V and a warning if expected counts are small.
    Result keys: table, chi2, p, dof, cramers_v, min_expected."""
    from scipy import stats

    a, b = _col(df, a, "a"), _col(df, b, "b")
    d = df[[a, b]].dropna()
    tab = pd.crosstab(d[a], d[b])
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        raise ValueError("both columns need at least 2 distinct values")
    chi2, p, dof, expected = stats.chi2_contingency(tab)
    n, k = int(tab.to_numpy().sum()), min(tab.shape) - 1
    v = float(np.sqrt(chi2 / (n * k)))
    low = float((expected < 5).mean())
    print(f"{a} x {b} (n={n}):")
    print(tab.to_string())
    sig = "significant" if p < alpha else "not significant"
    print(f"  Chi-square test of independence: chi2={chi2:.4f}, dof={dof}, p={_pfmt(p)} -> {sig} at {alpha}" + (" (Yates-corrected, 2x2)" if dof == 1 else ""))
    print(f"  Effect size: Cramer's V = {v:.3f} ({_label(v, (0.1, 0.3, 0.5), ('negligible', 'small', 'medium', 'large'))})")
    if low > 0.2:
        print(f"  WARNING: {low:.0%} of expected counts are below 5; merge rare categories or use an exact test")
    return Result(table=tab, chi2=float(chi2), p=float(p), dof=int(dof), cramers_v=v, min_expected=float(expected.min()))


def compare_proportions(df, flag, group, a, b=None, positive=None, alpha=0.05):
    """Two-proportion z-test (pooled standard error, two-sided) of the share of ROWS where `flag` is true
    (flag==positive, or Yes/True/1) in group `a` versus group `b` (b=None means all other groups).
    Result keys: z, p, p1, p2, n1, n2, x1, x2, diff, ci, relative_risk, fisher_p (only if counts are small)."""
    from scipy import stats

    flag, group = _col(df, flag, "flag"), _col(df, group, "group")
    d = df[[flag, group]].dropna()
    values = sorted(d[group].unique(), key=str)
    if a not in values:
        raise ValueError(f"{a!r} is not a value of '{group}'. Available: {values[:12]}")
    if isinstance(b, str) and b not in values and b.strip().lower() in {"other", "others", "rest", "all", "all other", "all others", "remaining"}:
        b = None  # intent is clear: everything except a
    if b is not None and b not in values:
        raise ValueError(f"{b!r} is not a value of '{group}'. Available: {values[:12]}. To compare against all other groups, leave b out.")
    hit = _binary(d[flag], positive)
    g1 = d[group] == a
    g2 = (d[group] != a) if b is None else (d[group] == b)
    n1, n2 = int(g1.sum()), int(g2.sum())
    if n1 < 2 or n2 < 2:
        raise ValueError(f"each group needs at least 2 rows (got {n1} and {n2}); available {group} values: {sorted(d[group].unique(), key=str)[:12]}")
    x1, x2 = int(hit[g1].sum()), int(hit[g2].sum())
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se0 = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se0 == 0:
        raise ValueError("no variation: every row has the same flag value")
    z = (p1 - p2) / se0
    p = float(2 * stats.norm.sf(abs(z)))  # sf, not 1 - cdf: stays accurate for tiny p
    se1 = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    crit = stats.norm.ppf(1 - alpha / 2)
    diff = p1 - p2
    ci = (float(diff - crit * se1), float(diff + crit * se1))
    rr = float(p1 / p2) if p2 > 0 else float("inf")
    fisher = float(stats.fisher_exact([[x1, n1 - x1], [x2, n2 - x2]])[1]) if min(n1 * pooled, n1 * (1 - pooled), n2 * pooled, n2 * (1 - pooled)) < 5 else None
    other = "all other groups" if b is None else str(b)
    print(f"Share of rows with {flag} = true: {a}: {x1}/{n1} = {p1:.2%}; {other}: {x2}/{n2} = {p2:.2%}")
    print(f"  Two-proportion z-test (pooled): z={z:.4f}, p={_pfmt(p)} -> {'significant' if p < alpha else 'not significant'} at {alpha}")
    print(f"  Difference ({a} - {other}) = {diff * 100:.2f} percentage points, {int((1 - alpha) * 100)}% CI [{ci[0] * 100:.2f}, {ci[1] * 100:.2f}]; relative risk = {rr:.2f}")
    if fisher is not None:
        print(f"  WARNING: small expected counts, the normal approximation is unreliable; Fisher exact p={_pfmt(fisher)}")
    return Result(z=float(z), p=p, p1=p1, p2=p2, n1=n1, n2=n2, x1=x1, x2=x2, diff=float(diff), ci=ci, relative_risk=rr, fisher_p=fisher)


# ------------------------------------------------------------------- time trend
def trend_forecast(df, date, value, freq="M", agg="sum", periods=1):
    """Aggregate `value` per period (freq: D, W, M, Q, Y), number the periods 0,1,2,... in time order,
    fit a least-squares straight line and forecast the next `periods` period(s).
    Periods with no rows count as 0 for sum/count. Ignores seasonality - say so when reporting.
    Result keys: slope, intercept, r2, p_slope, forecast (first next period), forecast_table, series, best, worst, n_periods."""
    from scipy import stats

    date, value = _col(df, date, "date"), _col(df, value, "value")
    freq = str(freq).upper()
    if freq not in {"D", "W", "M", "Q", "Y"}:
        raise ValueError("freq must be one of 'D', 'W', 'M', 'Q', 'Y'")
    d = df[[date, value]].dropna()
    try:
        when = pd.to_datetime(d[date])
    except Exception as e:
        raise ValueError(f"column '{date}' could not be read as dates ({e}); parse it first, e.g. clean_columns(df, dayfirst=True)") from None
    g = d[value].astype(float).groupby(when.dt.to_period(freq)).agg(agg).sort_index()
    full = pd.period_range(g.index.min(), g.index.max(), freq=freq)
    empty = len(full) - len(g)
    g = g.reindex(full, fill_value=0.0 if agg in ("sum", "count") else np.nan).dropna()
    if len(g) < 3:
        raise ValueError("need at least 3 periods to fit a trend")
    pos = full.get_indexer(g.index).astype(float)  # 0,1,2,... in time order (gaps keep their place)
    y = g.to_numpy(float)
    lr = stats.linregress(pos, y)
    nxt = np.arange(len(full), len(full) + int(periods))
    fvals = lr.intercept + lr.slope * nxt
    fper = pd.period_range(full[-1] + 1, periods=int(periods), freq=freq)
    unit = {"D": "day", "W": "week", "M": "month", "Q": "quarter", "Y": "year"}[freq]
    series = pd.DataFrame({"period": [str(p) for p in g.index], "index": pos.astype(int), "value": y})
    table = pd.DataFrame({"period": [str(p) for p in fper], "index": nxt, "forecast": fvals})
    best, worst = str(g.idxmax()), str(g.idxmin())
    adverb = {"D": "Daily", "W": "Weekly", "M": "Monthly", "Q": "Quarterly", "Y": "Yearly"}[freq]
    print(f"{adverb} {agg} of {value}: {len(g)} periods ({g.index[0]} to {g.index[-1]})" + (f"; {empty} empty periods counted as 0" if empty and agg in ("sum", "count") else ""))
    print(f"  Straight-line trend: slope = {_num(lr.slope)} per {unit} (p{_p(lr.pvalue)}, R2={lr.rvalue ** 2:.4f}) -> {'significant' if lr.pvalue < 0.05 else 'not significant'} at 0.05")
    print("  Forecast: " + ", ".join(f"{p}: {v:,.2f}" for p, v in zip(table["period"], fvals)) + "  (straight-line fit; ignores seasonality)")
    print(f"  Highest period: {best} ({g.max():,.2f}); lowest: {worst} ({g.min():,.2f})")
    return Result(slope=float(lr.slope), intercept=float(lr.intercept), r2=float(lr.rvalue**2), p_slope=float(lr.pvalue),
                  forecast=float(fvals[0]), forecast_table=table, series=series, best=best, worst=worst, n_periods=int(len(g)))


# ---------------------------------------------------------------------- charts
def quick_chart(df, x, y=None, hue=None, kind="bar", agg="sum", title=None, path="chart.png", top_n=None, figsize=(10, 6)):
    """Draw and save a chart. kind: bar, line, hist, scatter, box.
    bar/line aggregate y by x (agg: sum, mean, median, count, ...), hue splits into groups.
    hist uses only x. scatter needs x and y. box shows y by the groups in x.
    top_n keeps the largest top_n x-categories (bar/line). Returns the saved path."""
    import matplotlib.pyplot as plt

    kind = kind.lower()
    if kind not in {"bar", "line", "hist", "scatter", "box"}:
        raise ValueError("kind must be bar, line, hist, scatter or box")
    if kind in {"scatter", "box"} and y is None:
        raise ValueError(f"kind='{kind}' needs y")
    fig, ax = plt.subplots(figsize=figsize)
    default_title, ylabel = None, y

    if kind in {"bar", "line"}:
        keys = [x] + ([hue] if hue else [])
        g = df.groupby(keys)[y].agg(agg) if y else df.groupby(keys).size()
        ylabel = f"{agg} of {y}" if y else "count"
        default_title = f"{ylabel} by {x}" + (f" and {hue}" if hue else "")
        if hue:
            g = g.unstack(hue)
            if top_n:
                g = g.loc[g.sum(axis=1).nlargest(top_n).index].sort_index()
        else:
            if top_n:
                g = g.nlargest(top_n)
            elif kind == "bar" and not (pd.api.types.is_numeric_dtype(df[x]) or pd.api.types.is_datetime64_any_dtype(df[x])):
                g = g.sort_values(ascending=False)
        if kind == "bar" and isinstance(g.index, pd.DatetimeIndex):
            g.index = g.index.strftime("%Y-%m-%d")
        g.plot(kind=kind, ax=ax)
        if len(g) > 6:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        else:
            plt.setp(ax.get_xticklabels(), rotation=0)
        if np.nanmax(np.asarray(g, float)) >= 1000:
            from matplotlib.ticker import FuncFormatter

            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    elif kind == "hist":
        ylabel = "count"
        default_title = f"Distribution of {x}"
        if hue:
            for name, sub in df.groupby(hue):
                ax.hist(sub[x].dropna(), bins=30, alpha=0.5, label=str(name))
        else:
            ax.hist(df[x].dropna(), bins=30)
    elif kind == "scatter":
        default_title = f"{y} vs {x}"
        if hue:
            for name, sub in df.groupby(hue):
                ax.scatter(sub[x], sub[y], s=15, alpha=0.7, label=str(name))
        else:
            ax.scatter(df[x], df[y], s=15, alpha=0.7)
    else:  # box
        default_title = f"{y} by {x}"
        groups = [(str(k), v[y].dropna().to_numpy()) for k, v in df.groupby(x)]
        ax.boxplot([v for _, v in groups])
        ax.set_xticklabels([k for k, _ in groups])
        if len(groups) > 6:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_title(title or default_title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel or "")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(title=hue, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"chart saved: {path}")
    return str(path)


# ----------------------------------------------------------------- excel: reading
def _empty(v):
    return v is None or (isinstance(v, str) and not v.strip())


def _read_grid(ws):
    """All cell values of a worksheet as a list of rows. Vertically merged cells (e.g. a region name
    spanning several rows) are filled down; one-row merges (titles) are left alone."""
    rows = [list(r) for r in ws.iter_rows(min_row=1, min_col=1, values_only=True)]
    for rng in ws.merged_cells.ranges:
        if rng.max_row > rng.min_row and rng.min_row - 1 < len(rows):
            top = rows[rng.min_row - 1][rng.min_col - 1]
            for r in range(rng.min_row - 1, min(rng.max_row, len(rows))):
                for c in range(rng.min_col - 1, min(rng.max_col, len(rows[r]))):
                    rows[r][c] = top
    while rows and all(_empty(v) for v in rows[-1]):
        rows.pop()
    return rows


def _detect_header(rows):
    """Index of the header row: the first mostly-text row that is about as wide as the data below it."""
    counts = [sum(not _empty(v) for v in r) for r in rows]
    nonzero = sorted(c for c in counts if c)
    if not nonzero:
        return 0
    width = nonzero[len(nonzero) // 2]  # median width of a typical row
    for i, r in enumerate(rows[:40]):
        cells = [v for v in r if not _empty(v)]
        if len(cells) >= max(2, 0.7 * width) and sum(isinstance(v, str) for v in cells) >= 0.8 * len(cells):
            if i + 1 < len(rows) and counts[i + 1] >= 0.5 * width:
                return i
    return 0


_TOTAL_ROW = re.compile(r"^\s*(grand\s+total|sub-?\s*total|total)\b", re.I)


def _table_from_grid(rows, header_row=None):
    h = _detect_header(rows) if header_row is None else header_row
    head, body = rows[h], [r for r in rows[h + 1:] if not all(_empty(v) for v in r)]
    keep = [j for j in range(len(head)) if not (_empty(head[j]) and all(_empty(r[j]) for r in body))]
    names, seen = [], {}
    for j in keep:
        n = re.sub(r"\s+", " ", str(head[j])).strip() if not _empty(head[j]) else f"col_{j + 1}"
        seen[n] = seen.get(n, 0) + 1
        names.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    df = pd.DataFrame([[r[j] for j in keep] for r in body], columns=names)
    df = df.replace(r"^\s*$", np.nan, regex=True)
    is_total = df.iloc[:, :2].apply(lambda col: col.map(lambda v: isinstance(v, str) and bool(_TOTAL_ROW.match(v)))).any(axis=1)
    dropped = int(is_total.sum())
    return df[~is_total].reset_index(drop=True), h, dropped


def _workbook_facts(path):
    from openpyxl import load_workbook

    if str(path).lower().endswith(".xls"):
        raise ValueError("old .xls files are not supported: open it in Excel and save as .xlsx first")
    return load_workbook(path, data_only=True), load_workbook(path)


def inspect_workbook(path, verbose=True, **unknown):
    """Describe every sheet of an Excel file: size, header row, merged cells, formulas, charts, and the
    detected columns. Use this first on an unfamiliar workbook. Result keys: text, sheets (list of dicts)."""
    if unknown:
        raise TypeError(f"inspect_workbook() got unexpected argument(s) {sorted(unknown)}. It accepts: path, verbose")
    wb, wbf = _workbook_facts(path)
    infos, lines = [], [f"{Path(path).name}: {len(wb.worksheets)} sheet(s)"]
    for ws in wb.worksheets:
        wsf = wbf[ws.title]
        rows = _read_grid(ws)
        formulas = uncached = 0
        for r, rf in zip(ws.iter_rows(min_row=1, max_row=5000), wsf.iter_rows(min_row=1, max_row=5000)):
            for c, cf in zip(r, rf):
                if cf.data_type == "f":
                    formulas += 1
                    uncached += c.value is None
        info = dict(name=ws.title, hidden=ws.sheet_state != "visible", rows=len(rows), merged=len(ws.merged_cells.ranges),
                    formulas=formulas, formulas_without_value=uncached, charts=len(wsf._charts))
        if rows:
            try:
                df, h, dropped = _table_from_grid(rows)
                info.update(header_row=h + 1, columns=list(df.columns), table_rows=len(df), total_rows=dropped)
            except Exception as e:
                info["note"] = f"could not read a table ({e})"
        infos.append(info)
        line = f"- '{ws.title}'{' (hidden)' if info['hidden'] else ''}:"
        if "header_row" in info:
            line += (f" table header on row {info['header_row']}, then {info['table_rows']} data rows (the sheet range has {info['rows']} rows "
                     f"including title lines); columns: {', '.join(map(str, info['columns']))[:230]}")
        else:
            line += f" {info['rows']} rows"
        extra = [f"{info['merged']} merged ranges" if info["merged"] else "", f"{formulas} formulas" if formulas else "",
                 f"{uncached} formulas never calculated (open and save in Excel to get values)" if uncached else "",
                 f"{info['charts']} charts" if info["charts"] else "", f"{info.get('total_rows')} Total/Subtotal row(s) are NOT part of the {info.get('table_rows')} data rows: read_excel_smart already excludes them, so never drop any more rows" if info.get("total_rows") else ""]
        lines.append(line + (" [" + "; ".join(e for e in extra if e) + "]" if any(extra) else ""))
    text = "\n".join(lines)
    if verbose:
        print(text)
    return Result(text=text, sheets=infos)


def read_excel_smart(path, sheet=None, header_row=None, verbose=True, sheet_name=None, **unknown):
    """Read a messy Excel sheet into a DataFrame: finds the header row below title/blank lines, fills
    vertically merged cells, drops empty rows/columns and Total/Subtotal rows (which would double count),
    and says what it did. sheet = name or position (default: the biggest sheet); sheet_name is accepted as the same thing
    (pandas style). header_row = Excel row number to force."""
    if unknown:
        raise TypeError(f"read_excel_smart() got unexpected argument(s) {sorted(unknown)}. It accepts: path, sheet (or sheet_name), header_row, verbose")
    if sheet is not None and sheet_name is not None and sheet != sheet_name:
        raise TypeError("pass either sheet or sheet_name, not both with different values")
    sheet = sheet_name if sheet is None else sheet
    wb, wbf = _workbook_facts(path)
    names = wb.sheetnames
    if sheet is None:
        sheet = max(names, key=lambda n: sum(1 for r in wb[n].iter_rows(values_only=True) for v in r if not _empty(v)))
        chosen_note = f" (biggest of {len(names)} sheets: {names})" if len(names) > 1 else ""
    else:
        chosen_note = ""
    if isinstance(sheet, int):
        sheet = names[sheet]
    if sheet not in names:
        raise KeyError(f"sheet '{sheet}' not found. Sheets: {names}")
    ws = wb[sheet]
    rows = _read_grid(ws)
    if not rows:
        raise ValueError(f"sheet '{sheet}' is empty")
    df, h, dropped = _table_from_grid(rows, None if header_row is None else header_row - 1)
    wsf = wbf[sheet]
    uncached = sum(1 for r, rf in zip(ws.iter_rows(min_row=1, max_row=5000), wsf.iter_rows(min_row=1, max_row=5000))
                   for c, cf in zip(r, rf) if cf.data_type == "f" and c.value is None)
    notes = [f"header found on Excel row {h + 1}" + (f" ({h} title/blank rows skipped)" if h else "")]
    merged = sum(1 for m in ws.merged_cells.ranges if m.max_row > m.min_row)
    if merged:
        notes.append(f"{merged} merged ranges filled down")
    if dropped:
        notes.append(f"dropped {dropped} Total/Subtotal row(s)")
    if uncached:
        notes.append(f"WARNING: {uncached} formula cells have no stored value (file never calculated by Excel): open it in Excel, save, and read again")
    df.attrs["notes"] = notes
    if verbose:
        print(f"read_excel_smart: sheet '{sheet}'{chosen_note}: {len(df)} rows x {df.shape[1]} columns; " + "; ".join(notes))
    return df


# ----------------------------------------------------------------- excel: writing
def save_excel(path, sheets, autofit=True, totals=False, formulas=None, charts=None, sheet_name="Data", formats=None):
    """Write one DataFrame or a dict {sheet_name: DataFrame} to a formatted .xlsx
    (bold header, frozen header row, filters, column widths, number formats).
    Name the sheet with sheet_name='Summary' (single DataFrame) or by passing {'Summary': df}; the default name is 'Data'.
    formats={'Margin %': '0.0%'} sets the Excel number format of columns by name; year columns are shown without a thousands separator.
    totals=True            adds a Total row with live =SUM() formulas for numeric columns.
    formulas={'Margin': '{Profit}/{Revenue}'}   adds a live-formula column (placeholders are column names;
                           a tuple ('{Profit}/{Revenue}', '0.0%') also sets the number format). For several sheets
                           use {'SheetName': {...}}; a flat dict applies to the first sheet.
    charts=[{'kind': 'bar', 'x': 'Country', 'y': 'Revenue', 'title': '...'}]   adds native, editable Excel charts.
                           kind: bar (vertical), barh, line, pie. y may be a list. Optional: sheet, anchor, width, height."""
    from openpyxl.chart import BarChart, LineChart, PieChart, Reference
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    if isinstance(sheets, pd.DataFrame):
        sheets = {sheet_name: sheets}
    path = Path(path)
    first = next(iter(sheets))
    if formulas and not all(isinstance(v, dict) for v in formulas.values()):
        formulas = {first: formulas}
    formulas, charts, n_formula, n_chart = formulas or {}, list(charts or []), 0, 0
    header_fill = PatternFill("solid", fgColor="1F3864")

    def style_header(cell):
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, frame in sheets.items():
            safe = re.sub(r"[\[\]:*?/\\]", "_", str(name))[:31] or "Sheet"
            keep_index = not isinstance(frame.index, pd.RangeIndex)
            frame.to_excel(xw, sheet_name=safe, index=keep_index)
            ws = xw.sheets[safe]
            flat = frame.reset_index() if keep_index else frame
            cols, nrows = [str(c) for c in flat.columns], len(flat)
            last = nrows + 1
            for cell in ws[1]:
                style_header(cell)
            ws.freeze_panes = "A2"
            for idx, col in enumerate(flat.columns, start=1):
                series, letter = flat[col], get_column_letter(idx)
                fmt = ("yyyy-mm-dd" if pd.api.types.is_datetime64_any_dtype(series) else "#,##0.00" if pd.api.types.is_float_dtype(series)
                       else "#,##0" if pd.api.types.is_integer_dtype(series) else None)
                if pd.api.types.is_integer_dtype(series) and re.fullmatch(r"(fiscal)?(year|yr)", str(col).lower()):
                    fmt = "0"  # 2011, not 2,011
                fmt = (formats or {}).get(str(col), fmt)
                if fmt:
                    for row in ws.iter_rows(min_row=2, max_row=last, min_col=idx, max_col=idx):
                        row[0].number_format = fmt
                if autofit:
                    longest = max([len(str(col))] + [len(str(v)) for v in series.head(500).tolist()])
                    if totals and pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                        longest = max(longest, len(f"{series.sum():,.2f}"))  # the Total row must fit too
                    ws.column_dimensions[letter].width = min(max(10, longest + 2), 50)

            for newcol, spec in formulas.get(name, {}).items():  # live formula columns
                template, fmt = spec if isinstance(spec, (tuple, list)) else (spec, None)
                if newcol in cols:
                    raise ValueError(f"column '{newcol}' already exists in sheet '{name}'; drop it first or choose another name (formula columns are added for you)")
                if not re.search(r"\{[^}]+\}", template):  # '=Revenue/Units' -> '{Revenue}/{Units}'
                    resolved = template
                    for col in sorted(cols, key=len, reverse=True):
                        resolved = re.sub(r"(?<![A-Za-z0-9_{])" + re.escape(col) + r"(?![A-Za-z0-9_}])", "{" + col + "}", resolved)
                    if resolved == template:
                        raise ValueError(f"formula '{template}' for '{newcol}' mentions no column. Write column names in braces, e.g. '{{{cols[0]}}}/{{{cols[-1]}}}'. Available: {cols}")
                    template = resolved
                if "!" not in template:  # leftover bare names would become #NAME? in Excel
                    body = re.sub(r"\{[^}]+\}", " ", template)
                    unknown = [m.group(0) for m in re.finditer(r"[A-Za-z_][A-Za-z_0-9.]*", body)
                               if body[m.end():m.end() + 1] != "(" and m.group(0).upper() not in {"TRUE", "FALSE"}
                               and not re.fullmatch(r"\$?[A-Za-z]{1,3}\$?\d+", m.group(0))]
                    if unknown:
                        raise ValueError(f"formula for '{newcol}' uses unknown name(s) {unknown}. Column names must match exactly. Available: {cols}")
                missing = [m for m in re.findall(r"\{([^}]+)\}", template) if m not in cols]
                if missing:
                    raise ValueError(f"formula for '{newcol}' uses unknown columns {missing}. Available: {cols}")
                cidx = len(cols) + 1
                style_header(ws.cell(1, cidx, newcol))
                for r in range(2, last + 1):
                    expr = re.sub(r"\{([^}]+)\}", lambda m: f"{get_column_letter(cols.index(m.group(1)) + 1)}{r}", template)
                    cell = ws.cell(r, cidx, "=" + expr.lstrip("="))
                    if fmt:
                        cell.number_format = fmt
                ws.column_dimensions[get_column_letter(cidx)].width = max(12, len(newcol) + 2)
                cols.append(newcol)
                n_formula += 1
            ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{last}"

            if totals and nrows:  # live SUM row under the data
                tr = last + 1
                ws.cell(tr, 1, "Total").font = Font(bold=True)
                for idx, col in enumerate(flat.columns, start=1):
                    if pd.api.types.is_numeric_dtype(flat[col]) and not pd.api.types.is_bool_dtype(flat[col]) and idx > 1:
                        L = get_column_letter(idx)
                        c = ws.cell(tr, idx, f"=SUM({L}2:{L}{last})")
                        c.font, c.number_format = Font(bold=True), ws.cell(2, idx).number_format

            for spec in [c for c in charts if c.get("sheet", first) in (name, safe)]:  # native charts
                kind = str(spec.get("kind", "bar")).lower()
                if kind not in {"bar", "barh", "line", "pie"}:
                    raise ValueError("chart kind must be bar, barh, line or pie")
                ycols = list(spec["y"]) if isinstance(spec["y"], (list, tuple)) else [spec["y"]]
                bad = [c for c in [spec["x"]] + ycols if c not in cols]
                if bad:
                    raise ValueError(f"chart columns {bad} not found in sheet '{name}'. Available: {cols}")
                if nrows > 60 and kind != "line":
                    raise ValueError(f"the chart would have {nrows} categories; aggregate first (e.g. groupby) or use kind='line'")
                ch = {"bar": BarChart, "barh": BarChart, "line": LineChart, "pie": PieChart}[kind]()
                if kind in ("bar", "barh"):
                    ch.type = "col" if kind == "bar" else "bar"
                for yc in ycols:
                    ch.add_data(Reference(ws, min_col=cols.index(yc) + 1, min_row=1, max_row=last), titles_from_data=True)
                ch.set_categories(Reference(ws, min_col=cols.index(spec["x"]) + 1, min_row=2, max_row=last))
                ch.title = spec.get("title") or f"{', '.join(ycols)} by {spec['x']}"
                if kind != "pie":
                    ch.x_axis.title, ch.y_axis.title = spec["x"], ", ".join(ycols)
                    ch.x_axis.delete = ch.y_axis.delete = False  # openpyxl 3.1 otherwise hides the axes
                ch.width, ch.height = spec.get("width", 18), spec.get("height", 9)
                ws.add_chart(ch, spec.get("anchor", f"{get_column_letter(len(cols) + 2)}2"))
                n_chart += 1
    bits = [f"{len(sheets)} sheet(s)"] + ([f"{n_chart} chart(s)"] if n_chart else []) + (["totals row"] if totals else []) + ([f"{n_formula} formula column(s)"] if n_formula else [])
    print(f"excel saved: {path} ({', '.join(bits)})")
    return str(path)
