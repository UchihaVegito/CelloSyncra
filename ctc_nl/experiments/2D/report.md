# Results

## 1. Segmentation / tracking (seq 02)

| dataset        |   ('DET_F1', 'cpsam_pooled') |   ('DET_F1', 'oracle') |   ('SEG', 'cpsam_pooled') |   ('SEG', 'oracle') |   ('official_DET', 'cpsam_pooled') |   ('official_SEG', 'cpsam_pooled') |   ('official_TRA', 'cpsam_pooled') |
|:---------------|-----------------------------:|-----------------------:|--------------------------:|--------------------:|-----------------------------------:|-----------------------------------:|-----------------------------------:|
| DIC-C2DH-HeLa  |                        0.941 |                      1 |                     0.848 |               0.965 |                              0.912 |                              0.848 |                              0.908 |
| Fluo-N2DH-SIM+ |                        0.633 |                      1 |                     0.41  |               1     |                              0.583 |                              0.41  |                              0.575 |
| Fluo-N2DL-HeLa |                        0.945 |                      1 |                     0.885 |               0.909 |                              0.977 |                              0.885 |                              0.972 |
| PhC-C2DH-U373  |                        0.934 |                      1 |                     0.876 |               0.888 |                              0.959 |                              0.876 |                              0.947 |

## Attribute classifier (seq 02, all cells)

| dataset        |   macro_F1 |   balanced_acc |   AUROC_dividing |   AUPRC_dividing |   prevalence_dividing |   F1_dividing |   F1_newborn |
|:---------------|-----------:|---------------:|-----------------:|-----------------:|----------------------:|--------------:|-------------:|
| DIC-C2DH-HeLa  |      0.667 |          0.669 |            0.928 |            0.48  |                 0.016 |         0.486 |        0.533 |
| Fluo-N2DH-SIM+ |      0.566 |          0.728 |            0.956 |            0.781 |                 0.027 |         0.559 |        0.261 |
| Fluo-N2DL-HeLa |      0.571 |          0.642 |            0.808 |            0.22  |                 0.017 |         0.176 |        0.587 |
| PhC-C2DH-U373  |      0.328 |          0.333 |            0.171 |            0.009 |                 0.014 |         0     |        0     |

## 2. Language benchmark (score = F1 for selections/divisions, exact for counts)

|                                                          |   n |   score |   exact |   answered |   steps |   sec |   count_rel_err |
|:---------------------------------------------------------|----:|--------:|--------:|-----------:|--------:|------:|----------------:|
| ('openai-Qwen_Qwen2.5-72B-Instruct-AWQ', 'cpsam_pooled') | 480 |   0.557 |   0.329 |          1 |   4.112 | 4.479 |           0.086 |
| ('openai-Qwen_Qwen2.5-72B-Instruct-AWQ', 'oracle')       | 480 |   0.997 |   0.996 |          1 |   4.104 | 4.208 |           0     |

### By dataset

|                                                          |   DIC-C2DH-HeLa |   Fluo-N2DH-SIM+ |   Fluo-N2DL-HeLa |   PhC-C2DH-U373 |
|:---------------------------------------------------------|----------------:|-----------------:|-----------------:|----------------:|
| ('openai-Qwen_Qwen2.5-72B-Instruct-AWQ', 'cpsam_pooled') |           0.733 |            0.423 |            0.498 |           0.574 |
| ('openai-Qwen_Qwen2.5-72B-Instruct-AWQ', 'oracle')       |           1     |            1     |            1     |           0.986 |

### By query type

|                                                          |   about_to_divide |   above_median |   box |   count_all |   count_region |   divisions |   superlative |   topk_region |
|:---------------------------------------------------------|------------------:|---------------:|------:|------------:|---------------:|------------:|--------------:|--------------:|
| ('openai-Qwen_Qwen2.5-72B-Instruct-AWQ', 'cpsam_pooled') |             0.204 |          0.808 | 0.921 |         0.2 |           0.55 |       0.467 |         0.508 |         0.797 |
| ('openai-Qwen_Qwen2.5-72B-Instruct-AWQ', 'oracle')       |             0.983 |          1     | 1     |         1   |           1    |       1     |         1     |         0.989 |

### By wording style (template / template paraphrase / LLM free paraphrase / hand-written)

|                                                          |   LLM free paraphrase |   template |   template paraphrase |
|:---------------------------------------------------------|----------------------:|-----------:|----------------------:|
| ('openai-Qwen_Qwen2.5-72B-Instruct-AWQ', 'cpsam_pooled') |                 0.554 |      0.558 |                 0.558 |
| ('openai-Qwen_Qwen2.5-72B-Instruct-AWQ', 'oracle')       |                 0.996 |      0.994 |                 1     |

## 3. Error attribution

|                                                          |   n |   oracle_score |   backend_score |   planner_loss |   segmentation_loss |
|:---------------------------------------------------------|----:|---------------:|----------------:|---------------:|--------------------:|
| ('openai-Qwen_Qwen2.5-72B-Instruct-AWQ', 'cpsam_pooled') | 480 |          0.997 |           0.557 |          0.003 |                0.44 |
