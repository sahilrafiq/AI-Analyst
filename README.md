# AI-Analyst

A **local, offline AI data-analyst agent**. You ask for an analysis in plain English; a small language model running on
your own computer (through [Ollama](https://ollama.com)) writes Python, runs it, reads the result or the error, fixes
its mistakes and answers. It covers the everyday work of a data analyst: pandas, statistics, charts, Excel, written
reports and Power BI preparation.

It was built and tested on a modest laptop (16 GB RAM, 4 GB GTX 1650). No cloud service and no API key is needed, and
your data never leaves the machine.

## 👨‍💻 Developer

**Sahil Rafiq**

* GitHub: [@sahilrafiq](https://github.com/sahilrafiq)
* LinkedIn: [sahil-rafiq](https://www.linkedin.com/in/sahil-rafiq/)
* Portfolio: [sahil_rafiq](https://sahilrafiq.vercel.app/)

> **Status: working prototype.** A small local model makes mistakes. This project is mostly about *catching and
> reducing* them: tested helper functions, automatic guards, and benchmarks with known answers. Please read
> [Limitations](#limitations-and-safety) before relying on any result.

## What you can ask it

- "Clean this file, profile it, compare revenue across countries, and save an Excel summary and a chart."
- "Test whether the return rate differs between product E and the rest." / "Forecast next month with a linear trend."
- "Read the *Summary* sheet of `report.xlsx` (it has title lines and a Total row) and total Revenue by region."
- "Save a workbook with a Total row of live `SUM` formulas, a margin column of live formulas and a bar chart."
- `/report My title` &rarr; a Word, HTML and Markdown report of the session, with automatic number checks.
- "Prepare this data for Power BI with Product, Location and Customer dimensions and revenue measures."

## How it works

```
you ask  ->  agent previews your files (columns, types, first rows, every Excel sheet)
         ->  model writes ONE python block
         ->  block runs in a persistent Jupyter kernel (variables survive between steps)
         ->  output or error goes back to the model  ->  fix / next step  ->  short final answer
```

Small models are unreliable at free-form statistics, so the design leans on three things:

1. **Tested helper functions** (`analyst_tools.py`) so the model calls `compare_groups(...)` instead of writing a
   statistics routine from scratch. Helpers print readable summaries and raise *clear* errors (for example "y must be a
   column NAME, got Series. Available columns: ...") instead of pandas internals.
2. **Guards in the agent loop** (`agent.py`), for example:
   - results containing `nan`/`inf`/`None`, or a p-value printed as exactly `0`, are rejected and the model is told why;
   - identical successful code is not re-run; the same error three times triggers "use a different approach";
   - a required `RESULT ...` line must actually be printed by code;
   - requested sheet and column names are checked against the Excel files that were created;
   - cutting rows off a table that `read_excel_smart` already cleaned is blocked (it deletes real data);
   - numbers in the final answer that never appeared in any output are flagged.
3. **Benchmarks with planted answers** (`bench_*.py`): synthetic data where the correct result is known, graded exactly.

## Requirements

- Windows 10 (developed and tested there; Windows 11, Linux and macOS are probably fine but untested).
- Python 3.11 or newer.
- [Ollama](https://ollama.com) with a local chat model. Default: `gemma4:e4b`; alternative tested: `qwen3.5:4b`.
  The agent uses plain chat, not native tool calling.
- Tested hardware: Intel i5-9300H, 16 GB RAM, GTX 1650 (4 GB) - about 10 tokens/s with the default model.
  A 4 GB GPU cannot hold larger models; more VRAM gives faster, stronger models.

## Choosing a model for your hardware

Only the first row below was measured by the author; the others come from published guides and community reports and
are suggestions to try. The benchmarks in this repo are the way to compare models on *your* machine:
`python bench_hard.py --models gemma4:e4b qwen3.5:9b`.

| GPU memory | Suggestion | Status |
|---|---|---|
| 4 GB (GTX 1650 laptop) | `gemma4:e4b` (partly on the CPU), about 10 tokens/s | measured: 9/9 on `bench_hard.py` |
| 8 GB (RTX 2060 Super / 3060 / 4060) | `gemma4:e4b` first, then try `qwen3.5:9b` | untested by the author; expected several times faster |
| 12-16 GB | `qwen3.5:9b` or `gemma4:12b` with a longer context | untested |
| 24 GB or more | 26-35B class models | untested |

Rules of thumb: a Q4 model needs about 0.5-0.7 GB per billion parameters plus room for the context, so keep
`--num-ctx` at 8192 on small cards; check `ollama ps` shows the model running 100% on the GPU (a split between CPU and
GPU is what makes answers slow); more parameters do not automatically mean better results here - on the same tasks
`gemma4:e4b` clearly beat `qwen3.5:4b`.

## Install

```powershell
git clone https://github.com/sahilrafiq/AI-Analyst.git
cd AI-Analyst
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
ollama pull gemma4:e4b
```

Optional: keep Ollama's models on another drive (set the variable, then fully restart Ollama and sign out/in or reboot):

```powershell
mkdir D:\ollama\models
setx OLLAMA_MODELS "D:\ollama\models"
```

## Quick start

```powershell
mkdir workspace
copy path\to\your_data.csv workspace\
python agent.py
```

Then type a request at the `you>` prompt. Inside the chat:

| Command | Effect |
|---|---|
| `/report [title]` | write `report_*.docx / .html / .md` into the workspace from this session's results |
| `/files` | show what the agent sees in the workspace folder |
| `/reset` | fresh Python session and conversation |
| `/quit` | leave |

Options: `--model gemma4:e4b`, `--workdir workspace`, `--num-ctx 8192`, `--max-steps 6`, `--confirm` (approve every
code block before it runs), `--no-helpers`.

Every run writes `workspace/transcript_<time>.md` (everything the model said, every output and error) and
`workspace/session_<time>.py` (the code that ran successfully). Use the transcript to see exactly why something failed.

## Helper library (`analyst_tools.py`)

| Function | What it does |
|---|---|
| `clean_columns(df, dayfirst=None)` | converts numbers/dates stored as text (`$1,234.50`, `10%`, `25/12/2025`); **detects day-first vs month-first** from the values and warns if the data contradicts a given choice |
| `profile(df)` | types, missing values, statistics, likely problems (infinite values, numbers stored as text) |
| `compare_groups(df, value, group)` | checks assumptions and picks Welch t / Mann-Whitney / Welch ANOVA / Kruskal; effect size, post-hoc pairs |
| `regress(df, y, X)` | OLS with coefficients, CIs, diagnostics, `.predict(new_df)`; text columns become categories |
| `crosstab_test`, `compare_proportions` | chi-square with Cramer's V; pooled two-proportion z-test (uses `.sf`, so tiny p-values do not underflow to 0) |
| `trend_forecast(df, date, value)` | per-period totals, straight-line trend, next-period forecast |
| `top_correlations`, `quick_chart` | strongest correlations; bar/line/hist/scatter/box charts |
| `inspect_workbook`, `read_excel_smart` | understand and read messy Excel sheets (title rows, merged cells, Total rows) |
| `save_excel(..., totals, formulas, charts)` | formatted workbooks with live `SUM` totals, live formula columns, native Excel charts |

`report.py` turns a Markdown report into Word/HTML and checks that every number in it exists in the analysis output
(it also flags numbers that appear next to a different label than the output used). The checks are appended to the
document itself.

## Power BI (`powerbi_tools.py`)

Power BI Desktop cannot be driven by a script, so this part prepares everything for you:

`build_powerbi_package(df, "powerbi_output", fact="Sales", date="Date", value="Revenue", dims={...})` writes a star
schema as CSV (fact + dimensions with integer keys + a complete Date table), Power Query (M) load scripts with column
types, DAX measures (total, last year, YoY %, YTD, MTD, month-over-month, running total, 3-month moving average, % of
total, rank), a relationships guide, **`expected_values.xlsx`** (the same numbers computed independently in pandas) and a
`README_PowerBI.md` with the clicks to make in Power BI Desktop. It cross-checks the parsed date against `Year` /
`Month` / `Day` columns if your data has them, because a swapped day/month order silently moves revenue between months.

## Benchmarks

The benchmark scripts generate their own synthetic data with planted answers and need a running Ollama:

| Script | Tasks | What it checks |
|---|---|---|
| `benchmark.py` | 3 | one-shot code generation (no retries) |
| `bench_agent.py` | 3 | the same tasks through the agent loop |
| `bench_hard.py` | 9 (`--tier 1/2/all`) | messy-data cleaning, ANOVA, regression, outliers, monthly trend, chi-square, proportion z-test, forecast, categorical regression |
| `bench_excel.py` | 4 | read a messy workbook, merged cells, write a workbook with live `SUM` + a native chart, formula column |

Results with `gemma4:e4b` on the hardware above (28 Oct - 3 Nov 2026, local runs):

| Benchmark | Result |
|---|---|
| `bench_hard.py` (9 tasks) | 9/9 in the four latest runs; 8/9 in the two before. The single failure was different each time |
| `bench_excel.py` (4 tasks) | 4/4 on the latest run (189 s). Across seven runs made while the helpers were being fixed it scored 2, 3, 3, 4, 3, 3, 4 out of 4; each failure exposed a bug that was fixed afterwards |
| `qwen3.5:4b`, first five hard tasks | 2/5 in two runs, so `gemma4:e4b` is the default |

Please read these numbers as indicative: a small local model is not deterministic, each configuration was run only a few times, and the helper
changes made after the last benchmark runs (automatic date-order detection, Power BI tools) are unit-tested but not
re-benchmarked. Re-run the scripts on your own machine to see your numbers.

## Tests

The repository has 85 automated tests that need no Ollama, no GPU and no extra packages: the model is replaced by
scripted replies and the Jupyter kernel by a stand-in.

```powershell
python -m unittest discover -s tests        # about 1 minute; pytest also works
```

| File | What it checks |
|---|---|
| `tests/test_analyst_tools.py` | helpers against scipy / numpy / hand calculations (tests, regression, forecasts), date-order detection, Excel reading and writing; formulas are recalculated in LibreOffice when it is installed (skipped otherwise) |
| `tests/test_agent.py` | the guards and the loop with scripted model replies: nan and `p=0` rejection, repeated code, required output lines, requested sheet/column names, blocked row-dropping, report verification notes |
| `tests/test_report.py` | Markdown to Word/HTML conversion and the number checks |
| `tests/test_powerbi_tools.py` | star schema (lossless when read back from the CSV files), Power Query and DAX text, expected values against a plain-Python reference |
| `tests/test_benchmarks.py` | the benchmarks themselves: correct solutions pass, wrong ones fail, the planted answers have not drifted |

Many tests replay failures found during development. To check that the tests can fail, seven of those earlier bugs
were re-introduced one at a time; each made the suite fail. Last run: all 85 pass on Windows 10 with Python 3.14 (1 skipped,
the LibreOffice formula check) and on Linux with Python 3.12. Please open an issue if something fails on your setup.

**Checked by hand:** in Power BI Desktop, yearly totals, last-year values and YoY % produced by the generated measures
matched `expected_values.xlsx` on a 113,036-row sales file. The monthly measures should still be checked by the user.
The M and DAX text itself is only structure-checked automatically; it cannot be executed outside Power BI.

## Limitations and safety

- **Code runs on your machine, with your permissions, without a sandbox.** Use `--confirm` if you want to approve each
  block, and do not point the agent at files or instructions from sources you do not trust.
- **A small model can be wrong.** The guards catch many failure types (nan, invented numbers, wrong labels, ignored
  instructions) but not wrong reasoning. Read reports and check important numbers before sharing them.
- Reports include automatic verification notes; if they list items, check each one.
- The agent does not remove duplicate rows unless you ask; the helpers say how many exist.
- Excel: `.xlsx` only (old `.xls` is not supported), one table per sheet, multi-level headers are not flattened,
  formulas that Excel never calculated have no stored value (the tools warn).
- The `IPKernelApp WARNING ... TCP without encryption` message is harmless: the kernel only talks to your own computer.

## Troubleshooting

- **Models download again / land on `C:`** - the `OLLAMA_MODELS` variable was not seen by the Ollama tray app. Restart
  Ollama (and sign out or reboot), then check `ollama list`.
- **`Error: ... id_ed25519 ... cannot find the file`** - run `ollama serve` once; it recreates the key file.
- **`ModuleNotFoundError`** - `pip install -r requirements.txt` inside the activated virtual environment.
- **Slow answers** - about 10 tokens/s is normal on a 4 GB GPU. Thinking is switched off to save time.

## Project layout

```
agent.py          the agent loop, guards and chat commands
analyst_tools.py  helper functions (statistics, cleaning, charts, Excel)
report.py         Markdown -> Word/HTML and the number checks
powerbi_tools.py  Power BI star schema, Power Query (M), DAX, expected values
benchmark.py  bench_agent.py  bench_hard.py  bench_excel.py     benchmarks with planted answers
tests/            automated tests (python -m unittest discover -s tests)
CHANGELOG.md
```

## Contributing

Issues and pull requests are welcome; please run the tests before opening a pull request. The most useful reports include the `transcript_*.md` of a failed run (remove any
private data first) and your hardware and model.

## License

MIT - see [LICENSE](LICENSE).

## Acknowledgements

[Ollama](https://ollama.com), [Jupyter](https://jupyter.org), pandas, SciPy, openpyxl, python-docx, and the open model
families used here (Gemma by Google, Qwen by Alibaba).

---

*Built with ❤️ by [Sahil Rafiq](https://github.com/sahilrafiq)*