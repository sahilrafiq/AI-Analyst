"""
agent.py - a small local data-analyst agent (Phase 3).

How it works
  1. You describe a task. The agent shows the model your data files (columns,
     types, first rows) so it doesn't have to guess.
  2. The model replies with a Python code block.
  3. The code runs in a PERSISTENT Jupyter kernel (variables survive between steps).
  4. The output or the error goes back to the model, which fixes the code or
     gives a final answer. Repeats up to --max-steps times.

Every cell that ran without error is saved to workspace/session_<time>.py,
so you always get a reproducible script of what the agent did.

Usage
    python agent.py                       # interactive, model gemma4:e4b (best in our tests)
    python agent.py --no-helpers          # without the analyst_tools helper functions
    python agent.py --model qwen3.5:4b
    python agent.py --confirm             # ask before running each code block

Put your data files (csv / xlsx) in the workspace folder (default: ./workspace).
Commands inside the chat:  /report [title] (Word + HTML + Markdown report)   /reset   /files   /quit

Every run also writes workspace/transcript_<time>.md with everything the model said and every
output/error - send it to whoever is helping you debug a failure.

Keep analyst_tools.py (and report.py for /report) in the same folder as this file.
Needs:  pip install pandas numpy scipy scikit-learn matplotlib seaborn openpyxl jupyterlab python-docx
NOTE: the code the model writes runs on YOUR computer with your permissions.
      This is not a sandbox. Use --confirm if you want to approve each block.
"""
import argparse
import json
import queue
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = """You are a careful data analyst agent working in a persistent Python session.
pandas (pd), numpy (np) and matplotlib.pyplot (plt) are already imported. @@PACKAGES@@ Variables persist between steps.

Rules:
- To run code, reply with exactly ONE ```python block and nothing else. You will get the output or the error back.
- Never do arithmetic in your head. Compute every number with code and print it.
- Check column names and dtypes before using them. Parse dates with pd.to_datetime.
- Save charts with plt.savefig('name.png'). Never call plt.show().
- Write Excel files with pandas (openpyxl engine). Avoid reset_index() clashes by checking column names first.
- read_excel_smart already removes title lines and Total/Subtotal rows (it says so in its output): its result holds only data rows. Never cut off its last rows (no .iloc[:-1], .head(-1)).
- Never drop rows with missing values unless the task says so (then only for the columns the task uses). Silent row loss changes every result.
- Use the exact file, sheet and column names the task gives. Do not rename columns the task named.
- If the task says to print a specific line, print it with code. Writing it only in your answer does not count.
- Do not create files that were not requested. Read your data from the original file, not from files you made earlier.
- Your printed answer must follow the requested format exactly (e.g. a month label like 2024-12, not a row number; a single number, not a table).
- In the final summary only restate what the output shows. Do not speculate about what went wrong unless an error appeared.
- Write numbers as plain text (2.7317e-08), never LaTeX.
- Compute normal/t probabilities with scipy's .sf() (e.g. 2*norm.sf(abs(z))), never 1 - cdf: it underflows to exactly 0.
- Sanity-check every result before you answer: nan/inf values, impossible numbers (ages over 120, negative counts) or a per-group figure identical to the overall figure all mean the analysis is WRONG. Fix it. Never draw conclusions from nan.
- If code fails, fix the cause and send the corrected full block.
- When the task is complete, reply with a short plain-text summary that states the key numbers and names any files you saved, and NO code block."""

HELPERS_PROMPT = """

Ready-made helpers are already imported (tested - prefer them over writing the same code yourself):
- clean_columns(df, dayfirst=None, rename=False) -> cleaned copy: numbers and dates stored as text converted. Day-first or month-first dates are detected from the values; pass dayfirst only if the user states the format. Column names stay unchanged unless rename=True
- profile(df) -> prints types, missing values, stats, data problems
- compare_groups(df, value, group, groups=None, test='auto') -> picks the right test (t-test, Mann-Whitney, ANOVA, Kruskal), prints stats and effect size. Force one with test='welch'/'mannwhitney'/'anova'/'kruskal'. Result has .t .d .p .effect_size
- regress(df, y, X, robust=False) -> (y and X are column NAMES, not data) OLS with diagnostics; X is a list of column names (text columns become categories). Result has .predict(new_df) for forecasts, .params (coefficient by name, e.g. result.params['Units'] or result.params['West']; text columns become categories named like Region[West]), .coef (full table with p-values), .r2
- crosstab_test(df, a, b) -> chi-square test of independence between two categorical columns, with Cramer's V. Result has .chi2 .p .dof .cramers_v
- compare_proportions(df, flag, group, a, b=None, positive='Yes') -> pooled two-proportion z-test of the share of ROWS with flag==positive in group a versus group b (b=None means all other groups). Result has .z .p .p1 .p2 .diff .ci
- trend_forecast(df, date, value, freq='M', agg='sum', periods=1) -> totals per period, numbered 0,1,2,... in time order, a fitted straight line and the forecast of the next period(s). Result has .slope .intercept .r2 .p_slope .forecast .series .best .worst
- top_correlations(df, n=8)
- quick_chart(df, x, y=None, hue=None, kind='bar', agg='sum', title=None, path='chart.png', top_n=None); kinds: bar, line, hist, scatter, box
- inspect_workbook(path) -> describes every sheet of an Excel file (header row, merged cells, formulas, columns); use it first on unfamiliar workbooks
- read_excel_smart(path, sheet=None) (sheet_name= also works) -> DataFrame from a messy sheet: finds the header under title rows, fills merged cells, drops Total/Subtotal rows (and says so). Use it instead of pd.read_excel for business workbooks
- save_excel(path, df_or_dict_of_dfs, totals=False, formulas=None, charts=None, sheet_name='Data') -> formatted workbook. Name the sheet with sheet_name='Summary' or {'Summary': df}. totals=True adds a Total row of live SUM formulas; formulas={'Margin': '{Profit}/{Revenue}'} adds a live-formula column (do not add that column to the DataFrame yourself); charts=[{'kind':'bar','x':'Country','y':'Revenue','title':'...'}] adds native Excel charts (kinds: bar, barh, line, pie). Aggregate with groupby BEFORE charting
Always start from clean_columns(pd.read_csv(file)): it already parses text like '$1,234.50', '10%' and dates correctly. Never parse such text yourself (replacing commas breaks numbers).
If the task asks for a specific test, format or file name, follow it exactly."""

REPORT_SYSTEM = "You write clear, honest data-analysis reports in Markdown for a business reader."

REPORT_INSTRUCTIONS = """Write a short report (300-500 words) in Markdown based ONLY on the analysis output below.

Use exactly these sections: ## Summary (3-4 bullets with the key numbers), ## Data, ## Method, ## Findings, ## Caveats and next steps.
Rules:
- Use only numbers that appear in the analysis output, written in full (no "1.2M" abbreviations).
- For statistical tests give the test name, the p-value and the effect size.
- Do not invent causes or facts that are not in the output.
- Attach every number to the exact column or group named next to it in the output.
- Copy test results exactly as the output states them (significant or not, p-value, effect size). If a result you need is not in the output, say it was not computed; never guess it.
- Insert charts where relevant as ![short caption](file.png). Only these chart files exist: """

SHEET_SPEC = re.compile(r"(?:sheet|tab|worksheet)\s+(?:named|called)\s+['\"\u201c\u2018]([^'\"\u201d\u2019]+)['\"\u201d\u2019]", re.I)
COLUMN_SPEC = re.compile(r"column\s+(?:named|called)\s+['\"\u201c\u2018]([^'\"\u201d\u2019]+)['\"\u201d\u2019]", re.I)
ROW_DROP = re.compile(r"\.iloc\[\s*:\s*-\d+|\.head\(\s*-\d+|\[\s*:\s*-\d+\s*\]|\.drop\([^)]*index\[\s*-\d+")
COLUMNS_LIST_SPEC = re.compile(r"columns?\s+(?:named|called)\s+(?![\'\"\u201c\u2018])([A-Za-z_]\w*(?:\s*(?:,|\band\b)\s*[A-Za-z_]\w*)*)", re.I)
_NAME_STOP = {"a", "an", "the", "one", "two", "its", "their", "with", "for", "and"}
REQUIRED_LINE = re.compile(r"Print exactly (?:one|1) line:\s*(\w+)", re.I)
NONE_RESULT = re.compile(r"RESULT[^\n]*=\s*(None|nan|inf)\b", re.I)


P_ZERO = re.compile(r"\bp(?:[-_ ]?value)?\s*[=:]\s*0(?:\.0+)?(?:e[+\-]?0+)?(?=[\s,;)\]]|$)", re.I)


def result_warning(out):
    """A warning for results that cannot be right (nan/None, or a p-value printed as exactly 0)."""
    if NAN_STAT.search(out) or NONE_RESULT.search(out):
        return ("WARNING: the output contains nan/inf/None in a result, so the analysis did NOT work. "
                "Find the cause (infinite or missing values, a wrong column or key name - print the available names - "
                "groups with too few rows) and fix it. Do not draw conclusions from it.")
    if P_ZERO.search(out):
        return ("WARNING: a p-value was printed as exactly 0. That is rounding or numerical underflow (for example from 1 - cdf). "
                "Recompute it with the survival function (scipy .sf, e.g. 2*norm.sf(abs(z))) and print it in scientific notation.")
    return None


def package_line():
    """Tell the model which optional packages really exist in this Python (the kernel uses the same one)."""
    import importlib.util as iu

    names = {"scipy": "scipy", "statsmodels": "statsmodels", "scikit-learn": "sklearn", "seaborn": "seaborn",
             "plotly": "plotly", "openpyxl": "openpyxl", "xlsxwriter": "xlsxwriter"}
    have = [n for n, m in names.items() if iu.find_spec(m)]
    miss = [n for n, m in names.items() if not iu.find_spec(m)]
    line = "Also installed: " + ", ".join(have) + "."
    if miss:
        line += " NOT installed (never import them, the import will fail): " + ", ".join(miss) + "."
    return line


POWERBI_PROMPT = """
- build_powerbi_package(df, outdir, fact='Sales', date='Date', value='Revenue', dims={'Product': ['Product', 'Sub_Category'], 'Location': ['Country', 'State']}, fiscal_start_month=1, rank_by=('Product', 'Product')) -> writes a Power BI star schema as CSV files, a Power Query (M) load script, DAX measures, relationships.md, expected_values.xlsx and README_PowerBI.md. Use it when asked to prepare data or measures for Power BI. df needs a real date column (use clean_columns first); each column goes into the fact table or into exactly one dim; the value column stays in the fact
- dax_measures(fact, value, ...) and expected_values(df, date, value, by=None) can also be used on their own"""

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
NAN_STAT = re.compile(r"(p[-_ ]?value|\bp\s*[=:]|f[-_ ]?stat\w*|statistic|\b[tfuhr]\s*[=:]|\bcoef\w*|\br2\b|effect)[^\n]*\b(nan|inf)\b", re.I)
CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)(?:```|$)", re.S)


# --------------------------------------------------------------------- helpers
def shorten_traceback(tb):
    """Keep the model's own failing line plus the final error message; drop library internals."""
    lines = [l for l in ANSI.sub("", tb).splitlines() if l.strip()]
    cells = [i for i, l in enumerate(lines) if re.match(r"(Cell In\[|Input In \[|File <ipython-input)", l)]
    if not cells:
        return "\n".join(lines[-8:])
    i, block = cells[-1], []
    for l in lines[i:i + 7]:
        if block and re.match(r"(File |Cell In\[|Input In \[)", l):
            break
        block.append(l)
    err_at = next((j for j in range(len(lines) - 1, i, -1) if re.match(r"[A-Za-z_.]*(Error|Exception|Interrupt|Exit)\b", lines[j])), len(lines) - 1)
    tail = [l for l in lines[err_at:err_at + 4] if l not in block]
    return "\n".join(block + (["..."] if tail else []) + tail)


def clip(text, n):
    text = text.strip()
    if len(text) <= n:
        return text
    half = n // 2
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


KEY_LINE = re.compile(r"RESULT|Test:|Effect size|Assumptions|Post-hoc|Mean difference|Diagnostics|WARNING|chart saved|excel saved", re.I)


def condense(text, limit):
    """Shorten long output WITHOUT cutting results: statistical result blocks are always kept whole,
    the remaining space is filled with the other lines in order."""
    text = text.strip()
    if len(text) <= limit:
        return text
    lines = text.splitlines()
    keep, i = set(), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("OLS:"):
            j = i
            while j < len(lines) - 1 and "Diagnostics:" not in lines[j]:
                j += 1
            keep.update(range(i, j + 1))
            i = j + 1
            continue
        if re.match(r"^\S.* by \S.*: .* vs ", ln):
            j = i + 1
            while j < len(lines) and lines[j].startswith("  "):
                j += 1
            keep.update(range(i, j))
            i = j
            continue
        if ln.startswith("Top ") and "correlations" in ln:
            keep.update(range(i, min(i + 12, len(lines))))
            i += 12
            continue
        if KEY_LINE.search(ln):
            keep.add(i)
        i += 1
    used = sum(min(len(lines[k]), 700) + 1 for k in keep)
    for k, ln in enumerate(lines):
        if k not in keep and used + len(ln) + 1 <= limit:
            keep.add(k)
            used += len(ln) + 1
    out, prev = [], -1
    for k in sorted(keep):
        if k != prev + 1:
            out.append("...")
        out.append(lines[k][:700])
        prev = k
    if prev != len(lines) - 1:
        out.append("...")
    return "\n".join(out)


def extract_code(reply):
    blocks = CODE_BLOCK.findall(reply)
    blocks = [b.strip() for b in blocks if b.strip()]
    return max(blocks, key=len) if blocks else None


def post(payload):
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def describe_workspace(workdir):
    """Short text description of csv/xlsx files so the model sees real columns and types."""
    import pandas as pd

    parts = []
    for p in sorted(Path(workdir).iterdir()):
        if not p.is_file() or p.suffix.lower() not in (".csv", ".xlsx", ".xls") or p.name.startswith("session_"):
            continue
        try:
            if p.suffix.lower() == ".csv":
                df = pd.read_csv(p, nrows=2000)
                head = f"{p.name} (csv)"
            else:
                try:
                    import analyst_tools as at  # same folder as this file

                    parts.append(clip(at.inspect_workbook(p, verbose=False).text, 900))
                    df = at.read_excel_smart(p, verbose=False)
                    head = f"{p.name} - biggest sheet as a table"
                except Exception:
                    df = pd.read_excel(p, nrows=2000)
                    head = f"{p.name} (excel, sheets: {pd.ExcelFile(p).sheet_names})"
            cols = ", ".join(f"{c}:{t}" for c, t in df.dtypes.astype(str).items())
            rows = "\n".join("    " + line for line in df.head(3).to_string().splitlines())
            parts.append(f"{head}\n  columns: {cols}\n  first rows:\n{rows}")
        except Exception as e:
            parts.append(f"{p.name}: could not preview ({e})")
    return clip("\n".join(parts) or "(no data files yet)", 2500)


# ---------------------------------------------------------------- python kernel
def helper_setup():
    here = str(Path(__file__).resolve().parent)
    return (
        BASE_SETUP
        + "import sys\n"
        + f"if {here!r} not in sys.path: sys.path.insert(0, {here!r})\n"
        + "from analyst_tools import *\n"
        + "try:\n    from powerbi_tools import *\nexcept ImportError:\n    pass\n"
    )


BASE_SETUP = (
    "import warnings; warnings.filterwarnings('ignore')\n"
    "import matplotlib; matplotlib.use('Agg')\n"
    "import matplotlib.pyplot as plt\n"
    "import pandas as pd, numpy as np\n"
)


class PythonSession:
    def __init__(self, workdir, setup=BASE_SETUP):
        try:
            from jupyter_client.manager import start_new_kernel
        except ImportError:
            sys.exit("Missing packages. Run:  pip install jupyterlab ipykernel")
        self.setup = setup
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.km, self.kc = start_new_kernel(kernel_name="python3", cwd=str(self.workdir), startup_timeout=90)
        _, self.setup_error = self.run(self.setup)

    def restart(self):
        self.km.restart_kernel(now=True)
        self.kc.wait_for_ready(timeout=90)
        _, self.setup_error = self.run(self.setup)

    def close(self):
        try:
            self.kc.stop_channels()
            self.km.shutdown_kernel(now=True)
        except Exception:
            pass

    def run(self, code, timeout=120):
        """Execute code. Returns (stdout_text, error_text_or_None)."""
        msg_id = self.kc.execute(code, store_history=True)
        out, err = [], None
        deadline = time.time() + timeout
        interrupted = False
        while True:
            now = time.time()
            if now > deadline:
                if interrupted:
                    err = err or f"Timed out after {timeout}s"
                    break
                self.km.interrupt_kernel()
                interrupted = True
                deadline = now + 10
            try:
                msg = self.kc.get_iopub_msg(timeout=1)
            except queue.Empty:
                continue
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            kind, c = msg["msg_type"], msg["content"]
            if kind == "stream":
                out.append(c["text"])
            elif kind in ("execute_result", "display_data"):
                out.append(c["data"].get("text/plain", "") + "\n")
            elif kind == "error":
                tail = shorten_traceback("\n".join(c.get("traceback", []))) or f"{c['ename']}: {c['evalue']}"
                err = (f"Timed out after {timeout}s and was interrupted.\n" if interrupted else "") + tail
            elif kind == "status" and c["execution_state"] == "idle":
                break
        try:
            self.kc.get_shell_msg(timeout=2)
        except queue.Empty:
            pass
        return "".join(out), err


# ------------------------------------------------------------------------ agent
class Agent:
    def __init__(self, model="gemma4:e4b", workdir="workspace", num_ctx=8192, confirm=False, max_steps=6, helpers=True):
        self.model, self.num_ctx, self.confirm, self.max_steps = model, num_ctx, confirm, max_steps
        self.workdir = Path(workdir).resolve()
        self.session = PythonSession(self.workdir, helper_setup() if helpers else BASE_SETUP)
        if helpers and self.session.setup_error:
            print(f"WARNING: helper functions could not be loaded, continuing without them:\n{self.session.setup_error}")
            helpers = False
            self.session.setup = BASE_SETUP
        self.helpers = helpers
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        self.started = time.time()
        self.log_path = self.workdir / f"session_{stamp}.py"
        self.transcript_path = self.workdir / f"transcript_{stamp}.md"
        self.eval_tokens, self.eval_secs = 0, 0.0
        self._clear_state()

    def _clear_state(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT.replace("@@PACKAGES@@", package_line()) + ((HELPERS_PROMPT + (POWERBI_PROMPT if (Path(__file__).resolve().parent / "powerbi_tools.py").exists() else "")) if self.helpers else "")}]
        self.history = []  # one dict per executed code block: code, stdout, error
        self.tasks = []
        self.last_steps = 0
        self._files_desc = None

    def reset(self):
        self.session.restart()
        self._clear_state()

    def close(self):
        self.session.close()

    def _tlog(self, heading, text):
        with open(self.transcript_path, "a", encoding="utf-8") as f:
            f.write(f"\n## {heading}\n\n{text}\n")

    def _xlsx_snapshot(self):
        return {p.name: p.stat().st_mtime_ns for p in self.workdir.glob("*.xlsx") if not p.name.startswith("~$")}

    def _spec_problems(self, task, before):
        """Check workbooks created or changed during this task (not the input files) against the sheet/column names the task asked for."""
        sheets_wanted, cols_wanted = SHEET_SPEC.findall(task), COLUMN_SPEC.findall(task)
        loose = [p for m in COLUMNS_LIST_SPEC.finditer(task) for p in re.split(r"\s*(?:,|\band\b)\s*", m.group(1)) if p and p.lower() not in _NAME_STOP]
        now = self._xlsx_snapshot()
        files = [self.workdir / n for n, m in now.items() if before.get(n) != m]
        if not (sheets_wanted or cols_wanted or loose) or not files:
            return []
        try:
            from openpyxl import load_workbook

            names, headers = set(), set()
            for p in files:
                for ws in load_workbook(p).worksheets:
                    names.add(ws.title)
                    for row in ws.iter_rows(min_row=1, max_row=3, values_only=True):
                        headers.update(str(v) for v in row if v is not None)
        except Exception:
            return []
        return ([f"no sheet named '{x}' (sheets found: {sorted(names)})" for x in sheets_wanted if x not in names]
                + [f"no column named '{x}'" for x in cols_wanted if x not in headers]
                + [f"no column named '{x}' (the task asked for exactly this name; do not rename it)" for x in loose
                   if x not in headers and not any(h.startswith(x + " ") for h in headers)])

    def _checked(self, reply):
        """Flag numbers in the final summary that the analysis output never produced."""
        try:
            from report import unsupported_numbers
        except ImportError:
            return reply
        src = "\n".join(h["stdout"] for h in self.history if not h["error"]) + "\n" + "\n".join(self.tasks)
        bad = unsupported_numbers(reply, src)
        if bad:
            reply += "\n\n[CHECK: these numbers in the summary do not appear in the analysis output: " + ", ".join(bad) + "]"
        return reply

    def _trim(self):
        # keep the prompt inside the context window: drop the oldest step pair first
        budget = int(self.num_ctx * 3 * 0.6)
        while sum(len(m["content"]) for m in self.messages) > budget and len(self.messages) > 4:
            del self.messages[2:4]

    def _chat(self, messages=None):
        if messages is None:
            self._trim()
            messages = self.messages
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "options": {"num_ctx": self.num_ctx, "temperature": 0.2},
            "messages": messages,
        }
        try:
            try:
                data = post(payload)
            except urllib.error.HTTPError as e:
                if e.code != 400:
                    raise
                payload.pop("think")  # model doesn't accept the think field
                data = post(payload)
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot reach Ollama at {OLLAMA_URL} ({e}). Is Ollama running?")
        self.eval_tokens += data.get("eval_count", 0)
        self.eval_secs += data.get("eval_duration", 0) / 1e9
        return data["message"]["content"]

    def solve(self, task, on_event=None):
        emit = on_event or (lambda kind, text: None)
        content = task
        desc = describe_workspace(self.workdir)
        if desc != self._files_desc:
            content += "\n\nFiles in the workspace folder:\n" + desc
            self._files_desc = desc
        self.messages.append({"role": "user", "content": content})
        self.tasks.append(task)
        self._tlog("TASK", task)

        m_req = REQUIRED_LINE.search(task)
        required, reminded, first_cell = (m_req.group(1) if m_req else None), False, len(self.history)
        spec_reminded, row_drop_warned, xlsx_before = False, False, self._xlsx_snapshot()
        last_key, repeats = None, 0
        last_ok_code, last_ok_out, same_code = None, "", 0
        for step in range(1, self.max_steps + 1):
            self.last_steps = step
            reply = self._chat()
            self.messages.append({"role": "assistant", "content": reply})
            code = extract_code(reply)
            if code is None:
                printed = any(required in h["stdout"] for h in self.history[first_cell:] if not h["error"] and not h.get("warning")) if required else True
                if not printed and not reminded:
                    reminded = True
                    self.messages.append({"role": "user", "content": f"The task asked you to print a line starting with {required}, but your code never printed it (writing it in your answer does not count). Run code that prints exactly that line."})
                    continue
                if not spec_reminded:
                    problems = self._spec_problems(task, xlsx_before)
                    if problems:
                        spec_reminded = True
                        self.messages.append({"role": "user", "content": "The workbook you created does not follow the task: " + "; ".join(problems)
                                              + ". Fix it (for example save_excel(path, df, sheet_name='Summary')) and finish again."})
                        continue
                self._tlog("final answer", reply)
                return self._checked(reply.strip())
            self._tlog(f"step {step}: model code", reply)

            if (not row_drop_warned and ROW_DROP.search(code)
                    and ("read_excel_smart" in code or any("read_excel_smart" in h["code"] for h in self.history[first_cell:]))):
                row_drop_warned = True  # block once: cutting rows off a table the helper already cleaned deletes real data
                self.messages.append({"role": "user", "content": "STOP, this code was NOT run. read_excel_smart already removed the title lines and any Total/Subtotal rows "
                                      "(its output said so), so its result contains only data rows. Cutting off the last rows would delete real data. "
                                      "Remove that line and send the code again."})
                continue
            if code == last_ok_code:
                same_code += 1
                if same_code >= 2:
                    return "Stopped: the model kept resending the same code. Last output:\n" + clip(last_ok_out, 1500)
                self.messages.append({"role": "user", "content": "You already ran exactly this code successfully (see the output above). Do not run it again. Reply now with the final plain-text answer and NO code block."})
                continue
            emit("code", code)
            if self.confirm and input("Run this code? [y/N] ").strip().lower() != "y":
                return "Stopped: you declined to run the code."

            out, err = self.session.run(code)
            self.history.append({"code": code, "stdout": out, "error": err})
            self._tlog(f"step {step}: {'ERROR' if err else 'output'}", clip(err or out, 3000) or "(no output)")
            if err:
                emit("error", err)
                key = err.strip().splitlines()[-1]
                repeats = repeats + 1 if key == last_key else 1
                last_key = key
                feedback = (
                    f"The code failed:\n{clip(err, 1500)}\n\nOutput before the error:\n{clip(out, 800) or '(none)'}\n\n"
                    "Fix the cause and reply with the corrected full code block."
                )
                if "No module named" in err:
                    feedback += "\nThat package is NOT installed and cannot be installed here. Never import it again; use pandas, numpy, scipy or the ready-made helpers."
                if repeats >= 3:
                    feedback += ("\nThis is the same error for the third time. ABANDON this approach: do not call the function that fails again. "
                                 "Solve the task a different way, for example with plain pandas, numpy or scipy code.")
                elif repeats >= 2:
                    feedback += (
                        "\nYou made the same error again. Do not resend similar code: first print df.dtypes and "
                        "df.head() (or the objects involved), then write simpler code."
                    )
            else:
                last_key, repeats = None, 0
                last_ok_code, last_ok_out = code, out
                warn = result_warning(out)
                if warn:
                    self.history[-1]["warning"] = warn
                    emit("error", warn[:90] + "...")
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(f"# --- cell {len(self.history)} ---\n{code}\n\n")
                emit("output", out)
                feedback = (
                    (self.history[-1]["warning"] + "\n\n" if self.history[-1].get("warning") else "")
                    + f"Output:\n{clip(out, 1500) or '(no output)'}\n\n"
                    "If the task is complete, reply with a short plain-text summary and NO code block. "
                    "Otherwise reply with the next code block."
                )
            self.messages.append({"role": "user", "content": feedback})
        return "Stopped: step limit reached without a final answer."


    def write_report(self, title="Analysis report"):
        """Ask the model to write a report from this session's results, then build .md, .docx and .html."""
        ok = [h for h in self.history if not h["error"] and not h.get("warning") and h["stdout"].strip()]
        if not ok:
            return "Nothing to report yet: run an analysis first."
        charts = sorted(p.name for p in self.workdir.glob("*.png") if p.stat().st_mtime >= self.started - 1)
        # newest outputs first, within a share of the context window
        budget, used, picked = int(self.num_ctx * 3 * 0.6), 0, []
        for h in reversed(ok):
            o = condense(h["stdout"], 3500)
            if used + len(o) > budget:
                break
            picked.append(o)
            used += len(o)
        analysis = "\n---\n".join(reversed(picked))
        tasks = "\n".join(f"- {clip(t, 300)}" for t in self.tasks[-5:])
        prompt = (
            REPORT_INSTRUCTIONS + (", ".join(charts) or "(none)") + "\n\nWhat the user asked for:\n" + tasks
            + "\n\nAnalysis output:\n" + analysis
        )
        md = self._chat([{"role": "system", "content": REPORT_SYSTEM}, {"role": "user", "content": prompt}]).strip()
        if md.startswith("```"):  # model wrapped the whole report in a code fence
            md = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", md).strip()
        md = re.sub(r"\A\s*#\s+[^\n]*\n", "", md, count=1).lstrip()  # the title comes from the /report argument
        self._tlog("report draft", md)

        stem = f"report_{datetime.now():%Y%m%d_%H%M%S}"
        try:
            from report import mislabeled_numbers, to_docx, to_html, unsupported_numbers
        except ImportError:
            (self.workdir / f"{stem}.md").write_text(md, encoding="utf-8")
            return f"Saved {stem}.md only: report.py was not found next to agent.py."

        source = "\n".join(h["stdout"] for h in ok)
        checks = []
        bad = unsupported_numbers(md, source)
        if bad:
            checks.append("Numbers not found in the analysis output: " + ", ".join(bad))
        checks += [f"Possible wrong label: {m}" for m in mislabeled_numbers(md, source)]
        if re.search(r"\bnan\b", md, re.I):
            checks.append("The text mentions 'nan': an analysis step failed.")
        if checks:  # the warnings travel with the document, not just the console
            md += ("\n\n## Verification notes (automatic)\n" + "\n".join(f"- {c}" for c in checks)
                   + "\n\nThese notes were generated automatically. Confirm each item against the analysis before sharing this report.")
        (self.workdir / f"{stem}.md").write_text(md, encoding="utf-8")

        made, notes, missing = [f"{stem}.md"], [], []
        try:
            missing = to_docx(md, self.workdir / f"{stem}.docx", title, self.workdir)
            made.append(f"{stem}.docx")
        except ImportError as e:  # python-docx not installed
            notes.append(str(e))
        html_missing = to_html(md, self.workdir / f"{stem}.html", title, self.workdir)
        made.append(f"{stem}.html")
        if missing or html_missing:
            notes.append("The report mentions chart files that do not exist: " + ", ".join(missing or html_missing))
        if checks:
            notes.append("CHECK BEFORE SHARING (also written at the end of the report):\n  - " + "\n  - ".join(checks))
        return "Report saved in the workspace folder: " + ", ".join(made) + ("\n" + "\n".join(notes) if notes else "")


# -------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--workdir", default="workspace")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--confirm", action="store_true", help="ask before running each code block")
    ap.add_argument("--no-helpers", action="store_true", help="do not load analyst_tools helper functions")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print(f"Starting Python session in {Path(args.workdir).resolve()} ...")
    agent = Agent(args.model, args.workdir, args.num_ctx, args.confirm, args.max_steps, helpers=not args.no_helpers)
    print(f"Model: {args.model} | context: {args.num_ctx} | helpers: {'on' if agent.helpers else 'off'} | commands: /report /reset /files /quit")
    print(f"Put csv/xlsx files in the workspace folder. Successful code is logged to {agent.log_path.name}")

    def show(kind, text):
        if kind == "code":
            print("\n--- code ---\n" + clip(text, 1500))
        elif kind == "output":
            print("--- output ---\n" + (clip(text, 800) or "(no output)"))
        elif kind == "error":
            print("--- error (the agent will try to fix it) ---\n" + clip(text, 600))

    try:
        while True:
            try:
                task = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not task:
                continue
            if task == "/quit":
                break
            if task == "/reset":
                agent.reset()
                print("Fresh session.")
                continue
            if task.startswith("/report"):
                try:
                    print(agent.write_report(task[7:].strip() or "Analysis report"))
                except RuntimeError as e:
                    print(f"\nError: {e}")
                continue
            if task == "/files":
                print(describe_workspace(agent.workdir))
                continue
            try:
                answer = agent.solve(task, on_event=show)
                print("\nagent> " + answer)
            except KeyboardInterrupt:
                agent.session.km.interrupt_kernel()
                print("\n(interrupted)")
            except RuntimeError as e:
                print(f"\nError: {e}")
    finally:
        agent.close()
        if agent.eval_secs:
            print(f"\nAverage generation speed this session: {agent.eval_tokens / agent.eval_secs:.1f} tokens/s")


if __name__ == "__main__":
    main()
