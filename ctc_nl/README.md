
# Natural-language cell analysis on time-lapse microscopy (2D+t and 3D+t)



Part of **CelloSyncra**. A user asks a free-text question about a microscopy movie

("Which 2 biggest cells are in the lower half of frame 12?"). An LLM planner answers it by

calling deterministic tools on segmented and tracked cells, so every number in the answer is

computed from masks, not guessed by the language model.



## Pipeline

| Stage | Component |

|---|---|

| Segmentation | One Cellpose-SAM generalist fine-tuned on sequence 01 of all datasets; the joint model learns from 2D frames plus XY/XZ/YZ slices of 3D volumes; 3D inference uses `do_3D` with anisotropy |

| Tracking | IoU + Hungarian linking, division detection; CTC `res_track.txt` |

| Measurements | 2D: area, perimeter, circularity, solidity, aspect ratio, intensity. 3D: volume, surface area, sphericity, solidity, depth |

| Classifier | ViT-S/16 (timm, ImageNet-21k→1k) on [image, image, mask] crops: interphase / dividing / newborn (2D) |

| Tools | set_frame, filter, rank, count, get_cell, track, divisions, submit_answer (+ depth regions and 3D boxes) |

| Planner | Qwen2.5-72B-Instruct-AWQ served locally with vLLM (no API key, no internet) |



## Evaluation (Cell Tracking Challenge, seq 01 = train, seq 02 = test)

1. Segmentation/tracking: SEG, DET, official TRA/DET/SEG (py-ctcmetrics)

2. Language benchmark: generated questions in template, paraphrase, LLM free-text and hand-written wording; scored by selection F1, count error, division P/R

3. Error attribution: the same questions on reference masks (oracle) vs the generalist



Datasets: DIC-C2DH-HeLa, Fluo-N2DH-SIM+, Fluo-N2DL-HeLa, PhC-C2DH-U373 (2D);

Fluo-C3DH-A549, Fluo-C3DH-A549-SIM, Fluo-C3DL-MDA231 (3D).



## Usage

```bash

python ctc_nl_pipeline.py selftest --out /tmp/ctc_selftest      # synthetic 2D+3D, CPU

bash run_hpc.sh setup && bash run_hpc.sh setup_qwen               # login node, once

CTC_DATA=<2D root>:<3D root> CTC_OUT=<output folder> sbatch run_hpc.sh full2d3d

```

Data (celltrackingchallenge.net), trained weights and results are not included.

