"""
bench_agent.py - same 3 tasks as benchmark.py, but through the agent loop,
so the model can read errors and retry (what you will actually use day to day).

Keep benchmark.py, agent.py and this file in the same folder.

    python bench_agent.py
    python bench_agent.py --models qwen3.5:4b qwen2.5:7b
    python bench_agent.py --num-ctx 4096 --max-steps 5
"""
import argparse
import time

import benchmark as b
from agent import Agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3.5:4b", "gemma4:e4b", "qwen2.5:7b"])
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--no-helpers", action="store_true", help="test without the analyst_tools helpers")
    args = ap.parse_args()

    b.check_imports()
    b.WORK.mkdir(exist_ok=True)
    df = b.make_data()
    stats_truth, monthly_truth = b.compute_truth(df)
    truth = {"stats": stats_truth, "excel": monthly_truth}

    rows = []
    for model in args.models:
        print(f"\n=== {model} (agent mode, helpers {'off' if args.no_helpers else 'on'}) ===")
        agent = Agent(model, workdir=b.WORK, num_ctx=args.num_ctx, max_steps=args.max_steps, helpers=not args.no_helpers)
        try:
            for task, prompt in b.TASKS.items():
                agent.reset()
                for f in b.OUTPUT_FILES:
                    (b.WORK / f).unlink(missing_ok=True)
                t0 = time.time()
                try:
                    answer = agent.solve(prompt)
                except Exception as e:
                    print(f"  {task:6} FAIL  agent error: {e}")
                    rows.append((model, task, False, 0, 0))
                    continue
                wall = time.time() - t0
                # newest successful output first, so the checker reads the final RESULT line
                stdout = "\n".join(reversed([h["stdout"] for h in agent.history if not h["error"]]))
                try:
                    ok, note = b.CHECKS[task](stdout, truth)
                except Exception as e:
                    ok, note = False, f"check error: {e}"
                failed_cells = sum(1 for h in agent.history if h["error"])
                safe = model.replace(":", "_").replace("/", "_")
                if ok and task == "chart":
                    (b.WORK / "chart1.png").replace(b.WORK / f"agent_{safe}__chart.png")
                print(
                    f"  {task:6} {'PASS' if ok else 'FAIL'}  steps={agent.last_steps}  "
                    f"errors fixed/hit={failed_cells}  {wall:4.0f}s  {note}"
                )
                rows.append((model, task, ok, agent.last_steps, wall))
        finally:
            speed = agent.eval_tokens / agent.eval_secs if agent.eval_secs else 0
            agent.close()
            print(f"  average speed: {speed:.1f} tok/s")

    print("\n=== SUMMARY (agent mode) ===")
    print(f"{'model':14} {'passed':8} {'avg steps':10} {'total time':10}")
    for model in args.models:
        rs = [r for r in rows if r[0] == model]
        if not rs:
            continue
        passed = sum(r[2] for r in rs)
        steps = sum(r[3] for r in rs) / len(rs)
        total = sum(r[4] for r in rs)
        print(f"{model:14} {passed}/{len(rs):<6} {steps:<10.1f} {total:.0f}s")
    print("\nCompare with the one-shot results from benchmark.py. Open the saved charts in bench_work to judge them by eye.")


if __name__ == "__main__":
    main()
