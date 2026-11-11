# Changelog

## 0.1.0 - first public release (pre-release)

A working prototype of a local, offline data-analyst agent built around Ollama and a Jupyter kernel.

### Added
- **Agent** (`agent.py`): previews your files, lets the model write Python, runs it in a persistent Jupyter kernel and feeds
  output and errors back. Chat commands `/report`, `/files`, `/reset`, `/quit`; options `--model`, `--workdir`,
  `--num-ctx`, `--max-steps`, `--confirm`, `--no-helpers`. A transcript and a script of the successful code are saved per run.
- **Guards** in the loop: rejects `nan`/`inf`/`None` and `p = 0` results, does not re-run identical code, switches approach
  after the same error three times, requires a requested `RESULT` line to be printed, checks requested sheet and column
  names in created workbooks, blocks cutting rows off cleaned tables, flags numbers in answers that no output produced.
- **Helper library** (`analyst_tools.py`): cleaning with day-first/month-first detection, profiling, group comparison
  (Welch / Mann-Whitney / Welch ANOVA / Kruskal with effect sizes and post-hoc), OLS with diagnostics and prediction,
  chi-square, proportion test, trend forecast, correlations, charts.
- **Excel**: read messy workbooks (title rows, merged cells, Total rows) and write formatted workbooks with live `SUM`
  totals, live formula columns and native charts.
- **Reports** (`report.py`): Word / HTML / Markdown reports with automatic checks that every number exists in the analysis
  output and sits next to the right label; the notes are written into the document.
- **Power BI** (`powerbi_tools.py`): star schema CSVs, Power Query (M) scripts, DAX measures, a relationships guide,
  `expected_values.xlsx` for cross-checking, and a README with the steps in Power BI Desktop.
- **Benchmarks** with planted answers: `benchmark.py`, `bench_agent.py`, `bench_hard.py` (9 tasks), `bench_excel.py` (4 tasks).
- **Tests**: 85 automated tests (`python -m unittest discover -s tests`), no model or GPU needed. All pass on Windows 10 with
  Python 3.14 (1 skipped: the LibreOffice formula check) and on Linux with Python 3.12.

### Measured (author's machine: i5-9300H, 16 GB RAM, GTX 1650 4 GB, `gemma4:e4b`)
- `bench_hard.py`: 9/9 in the four latest runs. `bench_excel.py`: 4/4 on the latest run. About 10 tokens/s.

### Known limitations
- Code runs on your machine without a sandbox. A small model can still be wrong; read reports before sharing them.
- DAX and Power Query text are structure-checked automatically but cannot be executed outside Power BI; yearly figures
  were confirmed by hand in Power BI Desktop, monthly figures still need checking by the user.
- `.xls` is not supported; multi-level Excel headers are not flattened. Developed and tested on Windows 10.
