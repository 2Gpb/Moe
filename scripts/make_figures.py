"""Figures for the report.

    uv run python scripts/make_figures.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap

from finmoe.selection import domain_specificity, select_by_threshold

OUT = Path("report/figures")
FIN = ["fpb", "fiqasa", "tfns", "headlines"]

# method family -> (colour, marker); colour carries identity, marker repeats it for print / colour-blind readers
FAMILY = {
    "baseline": ("#8a8984", "s"),
    "router": ("#eb6834", "v"),
    "lora_attn": ("#2a78d6", "o"),
    "lora_experts": ("#1baf7a", "D"),
    "esft": ("#eda100", "^"),
    "control": ("#e87ba4", "x"),
    "dense": ("#008300", "P"),
}
METHOD_FAMILY = {
    "base_0shot": "baseline", "base_2shot": "baseline", "router": "router", "router_aux": "router",
    "lora_attn": "lora_attn", "lora_attn_router": "lora_attn", "lora_experts": "lora_experts",
    "lora_experts_r1": "lora_experts", "esft_token": "esft", "esft_gate": "esft",
    "esft_random": "control", "esft_least": "control", "dense_lora_attn": "dense",
}
DIVERGING = LinearSegmentedColormap.from_list("div", ["#d95926", "#f3f2ee", "#2a78d6"])


def style():
    plt.rcParams.update({
        "font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2c2c2a", "xtick.color": "#5f5e5a", "ytick.color": "#5f5e5a",
        "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "lines.linewidth": 1.5,
        "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def weighted_routing(per_task: dict, key: str = "selection_ratio") -> torch.Tensor:
    w = torch.tensor([float(per_task[t]["tokens"]) for t in FIN])
    return sum(wi * per_task[t][key] for wi, t in zip(w / w.sum(), FIN))


def fig_domain_specificity():
    base = torch.load("results/routing/base_train.pt", weights_only=False)["tasks"]
    fin = weighted_routing(base)
    spec = domain_specificity(fin, base["sst2"]["selection_ratio"]).numpy()
    chosen = select_by_threshold(torch.load("results/routing/esft_stats.pt", weights_only=False)["selection_ratio"], 0.2)

    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    lim = np.abs(spec).max()
    im = ax.imshow(spec.T, aspect="auto", cmap=DIVERGING, vmin=-lim, vmax=lim, interpolation="nearest")
    for layer, experts in chosen.items():
        ax.scatter([layer] * len(experts), experts, s=6, facecolors="none", edgecolors="#2c2c2a", linewidths=0.6)
    ax.set_xlabel("layer")
    ax.set_ylabel("expert")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("log(p_fin / p_general)")
    ax.set_title("Expert selection on financial vs. general (SST-2) text; circles = ESFT-Token experts (p=0.2)", fontsize=8)
    fig.savefig(OUT / "domain_specificity.pdf")
    plt.close(fig)


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = (p + q) / 2
    kl = lambda a, b: (a * (a.clamp_min(1e-12) / b.clamp_min(1e-12)).log()).sum(-1)
    return (kl(p, m) + kl(q, m)) / 2


def fig_task_similarity():
    base = torch.load("results/routing/base_train.pt", weights_only=False)["tasks"]
    tasks = FIN + ["sst2"]
    names = ["FPB", "FiQA", "TFNS", "Headl.", "SST-2"]
    mat = np.array([[float(js_divergence(base[a]["selection_ratio"], base[b]["selection_ratio"]).mean()) for b in tasks] for a in tasks])
    fig, ax = plt.subplots(figsize=(2.9, 2.5))
    im = ax.imshow(mat, cmap=LinearSegmentedColormap.from_list("seq", ["#f3f2ee", "#184f95"]))
    ax.set_xticks(range(5), names, rotation=45)
    ax.set_yticks(range(5), names)
    ax.grid(False)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{1000 * mat[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if mat[i, j] > mat.max() * 0.6 else "#2c2c2a")
    ax.set_title("Routing JS divergence (x1e-3)", fontsize=8)
    fig.savefig(OUT / "task_similarity.pdf")
    plt.close(fig)


def fig_pareto():
    df = pd.read_csv("results/all_runs.csv")
    g = df.groupby("method").agg(f1=("mean_f1", "mean"), std=("mean_f1", "std"), params=("trainable", "first"))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    seen = set()
    for method, r in g.iterrows():
        fam = METHOD_FAMILY.get(method, "baseline")
        colour, marker = FAMILY[fam]
        x = max(r["params"], 1e4)  # frozen baselines are drawn at the left edge
        ax.errorbar(x, r["f1"], yerr=0 if np.isnan(r["std"]) else r["std"], fmt=marker, color=colour, ms=5,
                    elinewidth=1, capsize=2, label=fam.replace("_", "-") if fam not in seen else None)
        seen.add(fam)
        ax.annotate(method.replace("base_", "").replace("_", "-"), (x, r["f1"]), xytext=(3, 3), textcoords="offset points", fontsize=5.5, color="#5f5e5a")
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters (log; frozen = left edge)")
    ax.set_ylabel("mean test macro-F1 (%)")
    ax.legend(fontsize=6, loc="lower right")
    fig.savefig(OUT / "pareto.pdf")
    plt.close(fig)


def fig_routing_shift():
    base = torch.load("results/routing/base_test.pt", weights_only=False)["tasks"]
    ref = weighted_routing(base)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for method in ["router", "router_aux", "lora_attn", "lora_attn_router", "lora_experts", "esft_token"]:
        path = Path(f"results/runs/{method}_s0/routing_test.pt")
        if not path.exists():
            continue
        shift = js_divergence(torch.load(path, weights_only=False)["selection_ratio"], ref).numpy()
        colour, marker = FAMILY[METHOD_FAMILY[method]]
        ls = "--" if method in ("router_aux", "lora_attn_router") else "-"
        ax.plot(range(len(shift)), shift, ls, color=colour, marker=marker, ms=3, label=method.replace("_", "-"))
    ax.set_xlabel("layer")
    ax.set_ylabel("JS(base, fine-tuned)")
    ax.legend(fontsize=6)
    ax.set_title("Routing shift after fine-tuning (test set)", fontsize=8)
    fig.savefig(OUT / "routing_shift.pdf")
    plt.close(fig)


def fig_training_curves():
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for method in ["router", "lora_attn", "lora_experts", "esft_token", "esft_random"]:
        path = Path(f"results/runs/{method}_s0/log.jsonl")
        if not path.exists():
            continue
        rows = [json.loads(l) for l in open(path)]
        rows = [r for r in rows if "loss" in r]
        steps = np.array([r["step"] for r in rows])
        loss = pd.Series([r["loss"] for r in rows]).ewm(alpha=0.3).mean()
        colour, marker = FAMILY[METHOD_FAMILY[method]]
        ax.plot(steps, loss, color=colour, label=method.replace("_", "-"))
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("training loss (EMA)")
    ax.legend(fontsize=6)
    fig.savefig(OUT / "training_curves.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    fig_domain_specificity()
    fig_task_similarity()
    if Path("results/all_runs.csv").exists():
        fig_pareto()
    if Path("results/routing/base_test.pt").exists():
        fig_routing_shift()
    fig_training_curves()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
