"""Run a list of (config, seed) jobs one after another, skipping finished ones.

Light artefacts of every finished run (summary, logs, predictions, routing
statistics) are copied to results/runs/ so they can be committed; adapter
weights stay in runs/.

    uv run python scripts/run_grid.py --configs lora_attn router esft_token --seeds 0 1 2
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import shutil
import subprocess

LIGHT = ("summary.json", "config.yaml", "log.jsonl", "test_predictions.jsonl", "routing_test.pt", "esft_experts.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--results", default="results/runs")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    for seed in args.seeds:
        for name in args.configs:
            run = f"{name}_s{seed}"
            run_dir = Path(args.runs) / run
            if (run_dir / "summary.json").exists():
                print(f"skip {run}: already finished", flush=True)
            else:
                cmd = [sys.executable, "scripts/train.py", "--config", f"configs/{name}.yaml", "--seed", str(seed),
                       "--out", args.runs, "--name", run]
                if args.set:
                    cmd += ["--set", *args.set]
                print(" ".join(cmd), flush=True)
                if subprocess.run(cmd).returncode != 0:
                    print(f"FAILED {run}", flush=True)
                    continue
            dest = Path(args.results) / run
            dest.mkdir(parents=True, exist_ok=True)
            for f in LIGHT:
                if (run_dir / f).exists():
                    shutil.copy(run_dir / f, dest / f)


if __name__ == "__main__":
    main()
