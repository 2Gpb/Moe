"""Run a list of (config, seed) jobs one after another, skipping finished ones.

Light artefacts of every finished run (summary, logs, predictions, routing
statistics) are copied to results/runs/ so they can be committed; adapter
weights stay in runs/. A run whose results are already in results/runs/ is
skipped too, so a grid can be continued in a fresh environment (e.g. a new
Kaggle session) after committing the results of the previous one.

    uv run python scripts/run_grid.py --configs lora_attn router esft_token --seeds 0 1 2
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

LIGHT = ("summary.json", "config.yaml", "log.jsonl", "test_predictions.jsonl", "routing_test.pt", "esft_experts.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--results", default="results/runs")
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--deadline-hours", type=float, default=None, help="do not start new runs after this many hours")
    args = parser.parse_args()

    start = time.time()
    for seed in args.seeds:
        for name in args.configs:
            run = f"{name}_s{seed}"
            run_dir = Path(args.runs) / run
            dest = Path(args.results) / run
            if (run_dir / "summary.json").exists() or (dest / "summary.json").exists():
                print(f"skip {run}: already finished", flush=True)
            elif args.deadline_hours and time.time() - start > args.deadline_hours * 3600:
                print(f"deadline reached, not starting {run}", flush=True)
                continue
            else:
                cmd = [sys.executable, "scripts/train.py", "--config", f"configs/{name}.yaml", "--seed", str(seed),
                       "--out", args.runs, "--name", run]
                if args.set:
                    cmd += ["--set", *args.set]
                print(" ".join(cmd), flush=True)
                if subprocess.run(cmd).returncode != 0:
                    print(f"FAILED {run}", flush=True)
                    continue
            if (run_dir / "summary.json").exists():
                dest.mkdir(parents=True, exist_ok=True)
                for f in LIGHT:
                    if (run_dir / f).exists():
                        shutil.copy(run_dir / f, dest / f)


if __name__ == "__main__":
    main()
