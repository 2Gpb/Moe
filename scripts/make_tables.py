"""Aggregate finished runs into the result tables (markdown + LaTeX).

    uv run python scripts/make_tables.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
import re

import pandas as pd

TASKS = ["fpb", "fiqasa", "tfns", "headlines"]
TASK_NAMES = {"fpb": "FPB", "fiqasa": "FiQA-SA", "tfns": "TFNS", "headlines": "Headlines"}
METHODS = {
    "base_0shot": "Zero-shot",
    "base_2shot": "Few-shot (2/label)",
    "router": "Router only",
    "lora_attn": "LoRA-Attn (r=16)",
    "lora_experts_r1": "LoRA-Experts (r=1)",
    "esft_token": "ESFT-Token",
    "esft_random": "ESFT-Random (ctrl)",
    "esft_specific": "ESFT-Specific (ours)",
    "dense_lora_attn": "Dense Qwen2.5-0.5B LoRA-Attn",
}


def load_runs(root: Path) -> pd.DataFrame:
    rows = []
    for summary in sorted(root.glob("*/summary.json")):
        m = re.match(r"(.+)_s(\d+)$", summary.parent.name)
        if not m:
            continue
        s = json.load(open(summary))
        row = {"method": m.group(1), "seed": int(m.group(2)), "trainable": s["trainable_params"],
               "total": s["total_params"], "train_min": s["train_seconds"] / 60, "peak_gb": s["peak_memory_gb"],
               "ms_per_example": s["test_ms_per_example"], "device": s["device"]}
        for t in TASKS + ["mean"]:
            if t in s["test"]:
                row[f"{t}_f1"] = 100 * s["test"][t]["macro_f1"]
                row[f"{t}_acc"] = 100 * s["test"][t]["accuracy"]
        rows.append(row)
    return pd.DataFrame(rows)


def load_baselines(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("base_*shot_test.json")):
        r = json.load(open(path))
        row = {"method": r["name"], "seed": 0, "trainable": 0, "ms_per_example": r["ms_per_example"], "device": r["device"]}
        for t in TASKS + ["mean"]:
            if t in r["metrics"]:
                row[f"{t}_f1"] = 100 * r["metrics"][t]["macro_f1"]
                row[f"{t}_acc"] = 100 * r["metrics"][t]["accuracy"]
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(mean: float, std: float, n: int) -> str:
    if pd.isna(mean):
        return "--"
    return f"{mean:.1f}" if n < 2 or pd.isna(std) else f"{mean:.1f} ± {std:.1f}"


def aggregate(df: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    cols = [f"{t}_{metric}" for t in TASKS + ["mean"]]
    g = df.groupby("method")
    mean, std, n = g[cols].mean(), g[cols].std(), g.size()
    out = pd.DataFrame(index=mean.index)
    out["Method"] = [METHODS.get(m, m) for m in mean.index]
    out["Trainable"] = g["trainable"].first()
    out["Seeds"] = n
    for c, t in zip(cols, TASKS + ["mean"]):
        out[TASK_NAMES.get(t, "Mean")] = [fmt(mean.loc[m, c], std.loc[m, c], n[m]) for m in mean.index]
    out["_mean"] = mean[f"mean_{metric}"]
    order = [m for m in METHODS if m in out.index] + [m for m in out.index if m not in METHODS]
    return out.loc[order]


def human(n: float) -> str:
    if n == 0:
        return "0"
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.2f}{unit}"
    return str(int(n))


def to_latex(table: pd.DataFrame, caption: str, label: str) -> str:
    cols = ["Method", "Params", "FPB", "FiQA-SA", "TFNS", "Headlines", "Mean"]
    lines = [r"\begin{table*}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{l r c c c c c}", r"\toprule", " & ".join(rf"\textbf{{{c}}}" for c in cols) + r" \\", r"\midrule"]
    best = table["_mean"].max()
    for _, r in table.iterrows():
        cells = [r["Method"], human(r["Trainable"])] + [str(r[c]).replace("±", r"$\pm$") for c in cols[2:]]
        if r["_mean"] == best:
            cells[-1] = rf"\textbf{{{cells[-1]}}}"
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def main():
    runs = load_runs(Path("results/runs"))
    base = load_baselines(Path("results/eval"))
    df = pd.concat([base, runs], ignore_index=True)
    out = Path("results")
    df.to_csv(out / "all_runs.csv", index=False)

    f1 = aggregate(df, "f1")
    acc = aggregate(df, "acc")
    with open(out / "main_table.md", "w") as f:
        f.write("## Test macro-F1 (%)\n\n")
        f.write(f1.drop(columns="_mean").assign(Trainable=f1["Trainable"].map(human)).to_markdown(index=False))
        f.write("\n\n## Test accuracy (%)\n\n")
        f.write(acc.drop(columns="_mean").assign(Trainable=acc["Trainable"].map(human)).to_markdown(index=False))
        f.write("\n")

    if not runs.empty:
        eff = runs.groupby("method")[["trainable", "train_min", "peak_gb", "ms_per_example"]].mean()
        eff.index = [METHODS.get(m, m) for m in eff.index]
        eff["trainable"] = eff["trainable"].map(human)
        with open(out / "efficiency_table.md", "w") as f:
            f.write(eff.round(2).to_markdown() + "\n")

    tables = Path("report/tables")
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "main_f1.tex").write_text(to_latex(f1, "Test macro-F1 (\\%) on the four financial tasks. Mean $\\pm$ std over seeds where available.", "tab:main"))
    (tables / "main_acc.tex").write_text(to_latex(acc, "Test accuracy (\\%).", "tab:acc"))
    print(open(out / "main_table.md").read())


if __name__ == "__main__":
    main()
