|                              | trainable   |   train_min |   peak_gb |   ms_per_example |
|:-----------------------------|:------------|------------:|----------:|-----------------:|
| Dense Qwen2.5-0.5B LoRA-Attn | 2.16M       |        2.47 |      5.46 |             5.18 |
| ESFT-Random (ctrl)           | 128.97M     |        6.28 |     11.51 |            12.53 |
| ESFT-Specific (ours)         | 128.97M     |        6.24 |     11.49 |            12.43 |
| ESFT-Token                   | 128.97M     |        6.91 |     11.95 |            13.25 |
| LoRA-Attn (r=16)             | 2.75M       |        5.03 |     10.21 |             9.51 |
| LoRA-Experts (r=1)           | 2.75M       |       14.61 |     13.39 |            25.64 |
| Router only                  | 786.43K     |        4.69 |      9.27 |             8.88 |
