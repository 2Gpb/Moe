# FinMoE: Parameter-Efficient Domain Adaptation of MoE Language Models for Financial NLP

This project studies *where* to spend a small parameter budget when adapting a sparse
Mixture-of-Experts (MoE) language model to financial text: in the attention layers, in
the experts, in the router, or in a hand-picked subset of experts chosen from routing
statistics (expert-specialised fine-tuning, ESFT).

Main model: [IBM Granite-3.1-1B-A400M](https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-base)
(1.33B parameters, 32 experts per layer, top-8 routing, ~400M active per token).
A dense model with a similar active size, Qwen2.5-0.5B, serves as a reference.

## Tasks

| Task | Source | Classes | Train / Val / Test |
|---|---|---|---|
| FPB | Financial PhraseBank (FLARE split) | neg / neu / pos | 3100 / 776 / 970 |
| FiQA-SA | FiQA 2018 task 1 (FLARE split) | neg / neu / pos | 750 / 188 / 235 |
| TFNS | Twitter Financial News Sentiment | neg / neu / pos | 2500 / 500 / 2388 |
| Headlines | Gold commodity news headlines (FLARE) | no / yes | 2500 / 500 / 2000 |
| SST-2 | Stanford Sentiment Treebank (validation) | neg / pos | – / – / 872 (routing reference only) |

All tasks are trained jointly (one adapter for the whole domain). Splits are frozen in
`data/splits/` (`configs/data.yaml` documents the subsampling; headlines that occur in
val/test are removed from train because FLARE asks several questions per headline).

Classification uses the language-model head directly: the prompt ends with `Answer:` and
the logits of one verbalizer token per class (` negative`, ` neutral`, ` positive`,
` No`, ` Yes`) are compared. The same scoring is used zero-shot and after fine-tuning.

## Adaptation methods

| Config | What is trained |
|---|---|
| `lora_attn` | LoRA (r=16) on q/k/v/o projections |
| `lora_experts`, `lora_experts_r1` | LoRA on every expert's gate/up and down projections (r=8, r=1) |
| `router`, `router_aux` | router weights only (with / without load-balancing loss) |
| `lora_attn_router` | attention LoRA + router |
| `esft_token`, `esft_gate` | full update of the experts that cover 20% of the domain's routing mass (ESFT) |
| `esft_specific` | same expert budget, experts ranked by domain specificity log(p_fin / p_general) |
| `esft_random`, `esft_least` | controls: same budget on random / least-used experts |
| `dense_lora_attn` | Qwen2.5-0.5B with attention LoRA |

## Layout

```
src/finmoe/
  data.py         tasks, prompts, fixed splits, collator
  moe.py          finds router/experts in any HF MoE model, routing statistics, load-balancing loss
  adapters.py     AdaptedExperts: expert LoRA and ESFT deltas on top of frozen 3D expert weights
  build.py        builds a model + adapters from a config; attention LoRA
  selection.py    expert selection for ESFT and the control conditions
  modeling.py     model loading, verbalizer head
  training.py     multi-task training loop, checkpointing, test evaluation
  evaluation.py   predictions and metrics
scripts/          prepare_data, train, run_grid, evaluate, analyze_routing, predict, make_tables, make_figures
configs/          one YAML per experiment (inherits from base.yaml)
notebooks/        exploration notebook and the Kaggle notebook for the multi-seed grid
results/          metrics, predictions and routing statistics of every run (committed)
report/           IEEE-format report (LaTeX)
tests/            unit tests and an end-to-end training smoke test on a tiny random MoE
```

## Reproducing

```bash
uv sync
uv run pytest
uv run python scripts/prepare_data.py                  # rebuilds data/splits (already committed)
uv run python scripts/evaluate.py --config configs/base.yaml --shots 0
uv run python scripts/analyze_routing.py               # routing statistics + ESFT relevance scores
uv run python scripts/run_grid.py --seeds 0 1 2 --configs lora_attn router esft_token esft_specific
uv run python scripts/make_tables.py
uv run python scripts/make_figures.py
```

One training run takes ~35-45 min on an Apple M1 Pro (MPS, bf16) and fits in 16 GB.
On Kaggle (2x T4, fp16) `notebooks/kaggle_grid.ipynb` runs two queues in parallel.

Inference with a trained adapter:

```bash
uv run python scripts/predict.py --run runs/lora_attn_s0 --task fpb "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn."
```

Report: `cd report && tectonic main.tex`.

## Notes

- Adapter weights are saved to `runs/<name>/adapter.pt` whenever the validation macro-F1
  improves; only the trainable tensors are stored, the frozen backbone comes from the Hub.
- On macOS with iCloud-synced folders the `.venv` files can get the `hidden` flag, which
  makes Python skip `.pth` files; the scripts therefore add `src/` to `sys.path` themselves.
