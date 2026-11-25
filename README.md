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
| `lora_experts_r1` | LoRA (r=1) on every expert's gate/up and down projections; same budget as `lora_attn` |
| `router` | router weights only |
| `esft_token` | full update of the experts that cover 20% of the domain's routing mass (ESFT-Token) |
| `esft_specific` | same expert budget, experts ranked by domain specificity log(p_fin / p_general) |
| `esft_random` | control: same budget on randomly chosen experts |
| `dense_lora_attn` | Qwen2.5-0.5B with attention LoRA |

## Results

Test macro-F1 (%), mean ± std over 3 seeds, Granite-3.1-1B-A400M (dense row: Qwen2.5-0.5B).
All runs on one Kaggle T4, fp16. Full tables: `results/main_table.md`, `results/efficiency_table.md`.

| Method | Trainable | FPB | FiQA-SA | TFNS | Headlines | Mean |
|---|---|---|---|---|---|---|
| Zero-shot | 0 | 37.3 | 51.9 | 33.5 | 45.5 | 42.0 |
| Few-shot (2/label) | 0 | 36.5 | 57.0 | 35.8 | 42.1 | 42.9 |
| Router only | 0.79M | 30.0 ± 3.5 | 21.3 ± 4.6 | 32.0 ± 2.1 | 49.6 ± 4.3 | 33.3 ± 1.3 |
| LoRA-Attn (r=16) | 2.75M | 84.2 ± 0.8 | 64.4 ± 1.6 | 82.1 ± 1.9 | 66.7 ± 14.1 | 74.3 ± 3.7 |
| LoRA-Experts (r=1) | 2.75M | 81.5 ± 1.5 | 57.6 ± 2.6 | 80.9 ± 0.1 | 43.5 ± 1.9 | 65.9 ± 1.5 |
| **ESFT-Token** | 129M | **85.7 ± 0.7** | **66.0 ± 2.7** | **84.4 ± 0.8** | 81.1 ± 1.9 | **79.3 ± 0.1** |
| ESFT-Random | 129M | 83.5 ± 0.4 | 61.3 ± 0.9 | 82.9 ± 0.6 | 73.8 ± 8.0 | 75.4 ± 2.3 |
| ESFT-Specific | 129M | 83.1 ± 0.9 | 59.6 ± 0.9 | 82.5 ± 0.7 | 57.9 ± 13.7 | 70.8 ± 3.3 |
| Dense LoRA-Attn | 2.16M | 84.1 ± 0.5 | 60.4 ± 0.7 | 83.5 ± 0.9 | **88.1 ± 0.9** | 79.0 ± 0.2 |

The three ESFT variants train the same expert budget; what differs is the share of routed
tokens that reaches the tuned experts (22.8% / 10.7% / 9.3%), and F1 follows it.

## Layout

```
src/finmoe/
  data.py         tasks, prompts, fixed splits, collator
  moe.py          finds router/experts in any HF MoE model, routing statistics
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

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), a GPU with 16 GB (a Kaggle T4 is enough).

```bash
git clone https://github.com/2Gpb/Moe.git && cd Moe
uv sync
uv run pytest
uv run python scripts/prepare_data.py                  # rebuilds data/splits (already committed)
uv run python scripts/evaluate.py --config configs/base.yaml --shots 0
uv run python scripts/analyze_routing.py               # routing statistics + ESFT relevance scores
uv run python scripts/run_grid.py --seeds 0 1 2 --configs lora_attn lora_experts_r1 router \
    esft_token esft_random esft_specific dense_lora_attn
uv run python scripts/make_tables.py
uv run python scripts/make_figures.py
```

All reported runs were trained on Kaggle (2x T4, fp16) with `notebooks/kaggle_grid.ipynb`, which runs
two queues in parallel; one run takes 2.5-15 min on a T4 and the full grid about an hour.

Inference with a trained adapter:

```bash
uv run python scripts/predict.py --run runs/lora_attn_s0 --task fpb "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn."
```

Report: `cd report && tectonic main.tex`.

## Notes

- Adapter weights are saved to `runs/<name>/adapter.pt` whenever the validation macro-F1
  improves; only the trainable tensors are stored, the frozen backbone comes from the Hub.
- Batches are padded with an ordinary token, not `<|endoftext|>`: in Granite that token gets
  huge activations, padded positions overflow fp16 and every gradient turns into NaN.
- On macOS with iCloud-synced folders the `.venv` files can get the `hidden` flag, which
  makes Python skip `.pth` files; the scripts therefore add `src/` to `sys.path` themselves.

## License

MIT, see `LICENSE`.
