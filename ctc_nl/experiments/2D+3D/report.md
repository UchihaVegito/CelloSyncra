# Results

## 1. Segmentation / tracking (seq 02)

| dataset            |   ('DET_F1', 'cpsam_joint') |   ('DET_F1', 'oracle') |   ('SEG', 'cpsam_joint') |   ('SEG', 'oracle') |   ('official_DET', 'cpsam_joint') |   ('official_SEG', 'cpsam_joint') |   ('official_TRA', 'cpsam_joint') |
|:-------------------|----------------------------:|-----------------------:|-------------------------:|--------------------:|----------------------------------:|----------------------------------:|----------------------------------:|
| DIC-C2DH-HeLa      |                       0.94  |                      1 |                    0.853 |               0.965 |                             0.911 |                             0.853 |                             0.908 |
| Fluo-C3DH-A549     |                       0.048 |                      1 |                    0.361 |               0.991 |                             0     |                             0.361 |                             0     |
| Fluo-C3DH-A549-SIM |                       0.192 |                      1 |                    0.67  |               1     |                             0.16  |                             0.67  |                             0.231 |
| Fluo-C3DL-MDA231   |                       0.384 |                      1 |                    0.104 |               0.752 |                             0.355 |                             0.072 |                             0.344 |
| Fluo-N2DH-SIM+     |                       0.572 |                      1 |                    0.363 |               1     |                             0.519 |                             0.363 |                             0.513 |
| Fluo-N2DL-HeLa     |                       0.956 |                      1 |                    0.877 |               0.909 |                             0.978 |                             0.877 |                             0.973 |
| PhC-C2DH-U373      |                       0.929 |                      1 |                    0.876 |               0.888 |                             0.962 |                             0.876 |                             0.951 |

## Attribute classifier (seq 02, all cells)

| dataset        |   macro_F1 |   balanced_acc |   AUROC_dividing |   AUPRC_dividing |   prevalence_dividing |   F1_dividing |   F1_newborn |
|:---------------|-----------:|---------------:|-----------------:|-----------------:|----------------------:|--------------:|-------------:|
| DIC-C2DH-HeLa  |      0.629 |          0.588 |            0.925 |            0.476 |                 0.016 |         0.357 |        0.545 |
| Fluo-N2DH-SIM+ |      0.612 |          0.635 |            0.965 |            0.712 |                 0.027 |         0.65  |        0.24  |
| Fluo-N2DL-HeLa |      0.564 |          0.654 |            0.809 |            0.17  |                 0.017 |         0.159 |        0.592 |
| PhC-C2DH-U373  |      0.328 |          0.333 |            0.389 |            0.012 |                 0.014 |         0     |        0     |

## 2. Language benchmark (score = F1 for selections/divisions, exact for counts)

|                                                         |   n |   score |   exact |   answered |   steps |   sec |   count_rel_err |
|:--------------------------------------------------------|----:|--------:|--------:|-----------:|--------:|------:|----------------:|
| ('openai-Qwen_Qwen2.5-72B-Instruct-AWQ', 'cpsam_joint') | 610 |   0.438 |   0.261 |          1 |   4.133 | 4.57  |           0.121 |
| ('openai-Qwen_Qwen2.5-72B-Instruct-AWQ', 'oracle')      | 610 |   0.997 |   0.997 |          1 |   4.082 | 4.122 |           0     |

### By dataset

|                                                         |   DIC-C2DH-HeLa |   Fluo-C3DL-MDA231 |   Fluo-N2DH-SIM+ |   Fluo-N2DL-HeLa |   PhC-C2DH-U373 |
|:--------------------------------------------------------|----------------:|-------------------:|-----------------:|-----------------:|----------------:|
| ('openai-Qwen_Qwen2.5-72B-Instruct-AWQ', 'cpsam_joint') |           0.702 |              0.023 |            0.425 |             0.52 |           0.554 |
| ('openai-Qwen_Qwen2.5-72B-Instruct-AWQ', 'oracle')      |           1     |              0.992 |            1     |             1    |           0.992 |

### By query type

|                                                         |   about_to_divide |   above_median |   box |   count_all |   count_region |   divisions |   superlative |   topk_region |   z_range |
|:--------------------------------------------------------|------------------:|---------------:|------:|------------:|---------------:|------------:|--------------:|--------------:|----------:|
| ('openai-Qwen_Qwen2.5-72B-Instruct-AWQ', 'cpsam_joint') |             0.178 |          0.629 | 0.763 |        0.12 |           0.48 |       0.386 |         0.387 |         0.647 |         0 |
| ('openai-Qwen_Qwen2.5-72B-Instruct-AWQ', 'oracle')      |             1     |          1     | 1     |        1    |           1    |       1     |         0.987 |         0.987 |         1 |

### By wording style (template / template paraphrase / LLM free paraphrase / hand-written)

|                                                         |   LLM free paraphrase |   template |   template paraphrase |
|:--------------------------------------------------------|----------------------:|-----------:|----------------------:|
| ('openai-Qwen_Qwen2.5-72B-Instruct-AWQ', 'cpsam_joint') |                 0.441 |      0.436 |                 0.436 |
| ('openai-Qwen_Qwen2.5-72B-Instruct-AWQ', 'oracle')      |                 0.99  |      1     |                 1     |

## 3. Error attribution

|                                                         |   n |   oracle_score |   backend_score |   planner_loss |   segmentation_loss |
|:--------------------------------------------------------|----:|---------------:|----------------:|---------------:|--------------------:|
| ('openai-Qwen_Qwen2.5-72B-Instruct-AWQ', 'cpsam_joint') | 610 |          0.997 |           0.438 |          0.003 |               0.559 |
