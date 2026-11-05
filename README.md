# CelloSyncra

**AI-Powered Cancer Screening and Early Detection Through Real-Time Single-Cell Analysis**

Every year, millions of cancer cases go undetected until it is too late. Current diagnostic methods are invasive, slow, and dependent on large clinical infrastructure that is out of reach for much of the world. We are changing that.

CelloSyncra lets anyone ask questions about microscopy videos of living cells in plain English, in 2D and 3D, and get answers that are computed from the cells themselves.

<p>
  <a href="ctc_nl/experiments/2D+3D/Codes"><b>Code</b></a> ·
  <a href="ctc_nl/experiments/2D+3D/report.md"><b>Results</b></a> ·
  <a href="ctc_nl/docs/architecture_detailed.pdf"><b>Architecture</b></a>
</p>

## Demo video

<!-- Drag CelloSyncra_demo.mp4 into this line while editing on github.com -->

## What it does

> *"Which 3 cells are the largest in the upper half of frame 12?"*
> *"Which cells in frame 16 are about to divide?"*
> *"How many cells are in the deeper half of the volume at time point 5?"*

1. **Find:** one Cellpose-SAM model, fine-tuned for both 2D and 3D, draws the outline of every cell.
2. **Track:** each cell is followed over time, including when it splits in two.
3. **Measure:** size, shape, position, brightness and depth (3D) of every cell are stored in a table.
4. **Ask:** an AI assistant, **Qwen2.5-72B** running locally on the university GPU, reads the question and calls measurement tools on that table. It never guesses numbers, and every answer is drawn on the image so it can be checked.

Everything runs on the GVSU Clipper cluster: no internet connection, no API costs, and the images never leave the lab.

## Results

Tested on 7 public datasets from the [Cell Tracking Challenge](https://celltrackingchallenge.net): 4 in 2D and 3 in 3D. Each dataset has two videos; we train on the first and test on the second.

| | |
|---|---|
| Test questions, checked automatically | **610** |
| Questions understood correctly (with ground-truth cell outlines) | **99.7%** |
| 3D questions understood correctly | **99%** |
| Average time per answer | **~4 s** |
| Average steps (tool calls) per answer | **~4** |

Answer score per test video, with the ground-truth outlines and with our AI's own outlines:

| Video | 2D / 3D | With ground truth | With our AI |
|---|---|---|---|
| DIC-C2DH-HeLa | 2D | 100% | 70% |
| Fluo-N2DH-SIM+ | 2D | 100% | 43% |
| Fluo-N2DL-HeLa | 2D | 100% | 52% |
| PhC-C2DH-U373 | 2D | 99% | 55% |
| Fluo-C3DL-MDA231 | 3D | 99% | 2% |

The assistant understands the questions; the remaining errors come from finding the cells, which is the focus of our next step, especially in 3D. Full tables and figures: [2D+3D results](ctc_nl/experiments/2D+3D/report.md) · [2D-only results](ctc_nl/experiments/2D/report.md).

## Repository

```
ctc_nl/
├── docs/                    architecture figure
└── experiments/
    ├── 2D/                  2D experiment: results, figures, recorded answers, Codes/
    └── 2D+3D/               2D+3D experiment: results, figures, recorded answers, Codes/
docs/                        demo website (GitHub Pages)
```

## Run it

The code runs on a SLURM GPU cluster (written for GVSU Clipper; adjust paths, account and partition in `run_hpc.sh`):

```bash
bash   run_hpc.sh setup                  # once, login node: Python packages
bash   run_hpc.sh setup_qwen             # once, login node: vLLM + Qwen2.5-72B weights
CTC_OUT=<results folder> sbatch run_hpc.sh full2d3d    # train and test the whole system
sbatch --time=8:00:00 run_hpc.sh web     # the web app: upload videos and ask questions
```

The Cell Tracking Challenge data is not included; download it from [celltrackingchallenge.net](https://celltrackingchallenge.net) and respect its terms of use.

## Team

**Ali Belhrak, Samir Iqbal, Kruttika Sutrave**
Department of Computer Science, Grand Valley State University

## Acknowledgements

Data: Cell Tracking Challenge (Maška et al., *Nature Methods* 20, 1010–1020, 2023). Segmentation: Cellpose-SAM. Assistant: Qwen2.5-72B-Instruct, served with vLLM. Computing: GVSU Clipper HPC.

## License

[Apache License 2.0](LICENSE)

https://github.com/user-attachments/assets/16791349-199e-4521-85db-68537322c034


