#!/bin/bash
# =============================================================================
# run_hpc.sh — SLURM driver for ctc_nl_pipeline.py on the GVSU HPC
#
#   bash  run_hpc.sh setup              # once, on a LOGIN node (needs internet)
#   sbatch run_hpc.sh selftest          # 5 min sanity check on synthetic data
#   sbatch run_hpc.sh gpu               # all GPU stages (segmentation + classifier)
#   bash  run_hpc.sh setup_qwen         # once, LOGIN node: installs vLLM + downloads Qwen2.5-72B-Instruct-AWQ (~41 GB)
#   sbatch run_hpc.sh qwen_bench        # Qwen (the system's LLM planner) answers the benchmark questions
#   bash  run_hpc.sh report             # tables in $CTC_OUT/results/report.md
#   sbatch run_hpc.sh <single stage>    # e.g. train_generalist, segment_generalist, eval_seg ...
#
# Protocol: seq 01 = train / model selection, seq 02 = test; *_test folders ignored.
# =============================================================================
#SBATCH --job-name=ctc_nl
#SBATCH --account=sutravek
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --exclude=g050,g051             # GPUs failed on 2026-11-01 (CUDA unknown error); remove once fixed
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=your_email@example.edu

set -euo pipefail

# ---------------------------------------------------------------- paths
PROJECT=/mnt/projects/sutravek_project/Ali_belhrak/SI_Project
export CTC_DATA=${CTC_DATA:-$PROJECT/CTC_Data/CTC_Data/2D}
export CTC_OUT=${CTC_OUT:-$PROJECT/CTC_Data/Thesis}
CODE_DIR=${CODE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}   # sbatch copies the script; use the submit dir
# Same Python env as your working Cellpose-SAM zero-shot / fine-tune jobs
# (module python/3.12.3 + cuda/12.9, venv .venv_cpsam). Override with VENV=... if needed.
VENV=${VENV:-$PROJECT/CTC_Data/Models/Cellpose-SAM/ZeroShot/.venv_cpsam}
PY="$VENV/bin/python $CODE_DIR/ctc_nl_pipeline.py"

# caches on project storage (home quotas are small)
export HF_HOME=${HF_HOME:-$PROJECT/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-$PROJECT/.cache/torch}
# reuse the cpsam weights already downloaded by your zero-shot run
export CPSAM_BASE=${CPSAM_BASE:-cpsam}   # starting weights for fine-tuning (same file you already have)
export CELLPOSE_LOCAL_MODELS_PATH=${CELLPOSE_LOCAL_MODELS_PATH:-$PROJECT/CTC_Data/Models/Cellpose-SAM/ZeroShot/cellpose_models}
export PIP_CACHE_DIR=$PROJECT/.cache/pip
mkdir -p "$CTC_OUT" "$CODE_DIR/logs" "$HF_HOME" "$TORCH_HOME" "$CELLPOSE_LOCAL_MODELS_PATH"

# LLM planner
# The system's LLM planner: Qwen2.5-72B-Instruct-AWQ, run locally with vLLM (see setup_qwen / start_qwen)
OPEN_LLM=${OPEN_LLM:-Qwen/Qwen2.5-72B-Instruct-AWQ}   # planner LLM (7B alternative: Qwen/Qwen2.5-7B-Instruct)

# Backups OUTSIDE Thesis/ (Clipper is not backed up): trained generalist + every finished results set
BACKUP=${BACKUP:-$PROJECT/CTC_Data/Models/Cellpose-SAM/Generalist_backup}


load_modules() {
  # same modules as your working Cellpose-SAM jobs
  if command -v module >/dev/null 2>&1; then
    module --force purge >/dev/null 2>&1 || true
    module load python/3.12.3 >/dev/null 2>&1 || true
    module load cuda/12.9     >/dev/null 2>&1 || true
  fi
  unset PYTHONPATH PYTHONHOME
  export PYTHONUNBUFFERED=1
  export MPLCONFIGDIR="$CTC_OUT/.mplcache"; mkdir -p "$MPLCONFIGDIR"
}

activate() {
  load_modules
  [ -x "$VENV/bin/python" ] || { echo "ERROR: $VENV not found" >&2; exit 1; }
  export PATH="$VENV/bin:$PATH"          # no `source activate`: safe under set -u
  "$VENV/bin/python" -c "
import importlib.util, sys
need = ['torch','cellpose','cellpose.train','tifffile','skimage','scipy','pandas','timm','sklearn']
bad = [m for m in need if importlib.util.find_spec(m.split('.')[0]) is None]
if bad: sys.exit('missing in venv: ' + ', '.join(bad) + '  -> run: bash run_hpc.sh setup')
import torch, cellpose; print('env OK | python', sys.version.split()[0], '| torch', torch.__version__, '| cellpose', cellpose.version)"
}

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------- stages
setup() {   # LOGIN node: needs internet. Adds ONLY the missing packages to the shared venv.
  load_modules
  [ -x "$VENV/bin/python" ] || { echo "ERROR: $VENV not found" >&2; exit 1; }
  PIP="$VENV/bin/pip"
  log "freezing current versions so torch/cellpose/numpy are never changed"
  "$PIP" freeze | grep -iE "^(torch|torchvision|cellpose|numpy|scipy|scikit-image|tifffile)==" > "$CTC_OUT/.venv_pins.txt" || true
  cat "$CTC_OUT/.venv_pins.txt"
  # extra packages for this pipeline; -c pins keep the existing Cellpose-SAM stack intact
  "$PIP" install -q -c "$CTC_OUT/.venv_pins.txt" timm scikit-learn tabulate openai \
                 py-ctcmetrics huggingface_hub pillow
  # pre-download ViT weights so compute nodes need no internet (cpsam weights already exist).
  # HF_HUB_OFFLINE may be set in your shell; lift it only for this download.
  env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE -u HF_DATASETS_OFFLINE "$VENV/bin/python" - <<'PYEOF' \
    || log "WARNING: ViT weights not downloaded - classifier would train from scratch"
import timm; timm.create_model("vit_small_patch16_224", pretrained=True); print("ViT weights cached")
PYEOF
  activate
  log "setup done. venv: $VENV"
}

selftest()          { $PY selftest --out "$CTC_OUT/selftest"; }
prepare()           { $PY prepare; }
# ---- Segmentation: ONE generalist Cellpose-SAM trained on seq 01 of ALL datasets ----
train_generalist()  {
  # reuse the backed-up generalist if Thesis/models was lost, instead of retraining
  local M="$CTC_OUT/models/cpsam/models/cpsam_ft_pooled"
  if [ ! -f "$M" ] && [ -f "$BACKUP/cpsam_ft_pooled" ]; then
    mkdir -p "$(dirname "$M")"; cp "$BACKUP/cpsam_ft_pooled" "$M"; log "restored generalist from $BACKUP"
  fi
  $PY train_cpsam --mode pooled --epochs 100 --lr 1e-5 --weight-decay 0.1
  mkdir -p "$BACKUP"
  if [ -f "$M" ] && ! cmp -s "$M" "$BACKUP/cpsam_ft_pooled" 2>/dev/null; then
    cp "$M" "$BACKUP/cpsam_ft_pooled"; log "generalist backed up -> $BACKUP/cpsam_ft_pooled"
  fi
}
backup_results() {   # snapshot of everything small (masks, bench, results, classifier) - not the 1.2 GB model
  mkdir -p "$BACKUP/results"
  local TS; TS=$(date +%Y%m%d_%H%M)
  ( cd "$CTC_OUT" && tar czf "$BACKUP/results/thesis_${TS}.tgz" \
      --exclude='models/cpsam' --exclude='models/_*' --exclude='.mplcache' \
      data_summary.json results bench seg models logs 2>/dev/null ) || true
  log "results snapshot -> $BACKUP/results/thesis_${TS}.tgz"
}
segment_generalist(){ $PY segment --backend cpsam_pooled; }        # default thresholds (as on new data)
segment_oracle()    { $PY segment --backend oracle; }
eval_seg()          { $PY eval_seg --official --backends oracle cpsam_pooled; }
train_cls()         { $PY train_cls --epochs 15 --lr 1e-4 --batch-size 64; }
eval_cls()          { $PY eval_cls; }
make_bench() {      # template questions; kept if they already exist (FORCE_BENCH=1 to regenerate)
  if [ -f "$CTC_OUT/bench/queries.jsonl" ] && [ "${FORCE_BENCH:-0}" != "1" ]; then
    log "keeping existing bench/queries.jsonl ($(wc -l < "$CTC_OUT/bench/queries.jsonl") questions)"; return 0; fi
  $PY make_bench --n-per-type 5 --seed 0
}

# Benchmark backends: GT masks (upper bound) and the generalist
BACKENDS=(oracle cpsam_pooled)

# ============================ PLANNER: Qwen2.5-72B-Instruct-AWQ (local, vLLM) ============================
# The single LLM of the system. Served on the job's GPU through an OpenAI-compatible endpoint;
# no API key, no internet, no cost. Install once with:  bash run_hpc.sh setup_qwen   (login node)
VLLM_VENV=${VLLM_VENV:-$PROJECT/.venv_vllm}
# one port per job: two LLM jobs on the same node must not reach each other's server
QWEN_PORT=${QWEN_PORT:-$((8100 + ${SLURM_JOB_ID:-0} % 800))}
QWEN_URL=http://localhost:$QWEN_PORT/v1
# Model switch (no code change): 72B is the default (24/24 on the quick test vs 0.71 F1 for 7B).
# Smaller, faster alternative (same family/parser), e.g. for a size comparison:
#   OPEN_LLM=Qwen/Qwen2.5-7B-Instruct bash run_hpc.sh setup_qwen               (login node, ~15 GB once)
#   OPEN_LLM=Qwen/Qwen2.5-7B-Instruct sbatch run_hpc.sh qwen_bench
# Results go to a separate file per model, so 7B and 72B can be compared side by side.
MODEL_TAG=${OPEN_LLM//\//_}
MODEL_PATH_FILE="$VLLM_VENV/model_path_${MODEL_TAG}.txt"
TOOL_PARSER=${TOOL_PARSER:-hermes}

setup_qwen() {   # LOGIN node (internet): separate venv for vLLM + download Qwen weights (~15 GB) once
  load_modules
  [ -x "$VLLM_VENV/bin/python" ] || python3 -m venv "$VLLM_VENV"
  "$VLLM_VENV/bin/pip" install -q --upgrade pip
  "$VLLM_VENV/bin/pip" install -q vllm huggingface_hub
  env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE "$VLLM_VENV/bin/python" -c "
from huggingface_hub import snapshot_download
p = snapshot_download('$OPEN_LLM'); print('model cached at', p)
open('$MODEL_PATH_FILE', 'w').write(p)"
  "$VLLM_VENV/bin/python" -c "import vllm; print('vLLM', vllm.__version__, 'OK')"
}

start_qwen() {   # start the local Qwen server on this job's GPU and wait until it answers
  [ -x "$VLLM_VENV/bin/python" ] || { echo "ERROR: run 'bash run_hpc.sh setup_qwen' on the login node first" >&2; exit 1; }
  if curl -s "$QWEN_URL/models" | grep -q "\"$OPEN_LLM\""; then log "server for $OPEN_LLM already running"; return 0; fi
  if curl -s "$QWEN_URL/models" >/dev/null; then echo "ERROR: port $QWEN_PORT serves another model, not $OPEN_LLM" >&2; exit 1; fi
  # find the weights of THIS model: a saved path is used only if it is the right model's folder
  local MDIR="models--${OPEN_LLM//\//--}" MPATH="" c
  for c in "$(cat "$MODEL_PATH_FILE" 2>/dev/null)" \
           $(ls -d "$PROJECT"/../.hf_cache/hub/$MDIR/snapshots/*/ "$HF_HOME"/hub/$MDIR/snapshots/*/ 2>/dev/null); do
    if [ -n "$c" ] && [[ "$c" == *"/$MDIR/"* ]] && [ -f "${c%/}/config.json" ]; then MPATH="${c%/}"; break; fi
  done
  [ -n "$MPATH" ] || { echo "ERROR: weights for $OPEN_LLM not found - run: OPEN_LLM=$OPEN_LLM bash run_hpc.sh setup_qwen" >&2; exit 1; }
  echo "$MPATH" > "$MODEL_PATH_FILE"
  local VLOG="$CODE_DIR/logs/vllm_${SLURM_JOB_ID:-local}.log"
  # Clipper fix: FlashInfer JIT-compiles kernels with the system GCC, but Python (miniconda) loads an
  # older libstdc++ -> "GLIBCXX_3.4.32 not found". Use vLLM's PyTorch sampler and the newer libstdc++.
  export VLLM_USE_FLASHINFER_SAMPLER=0
  local GXX_LIB; GXX_LIB=$(dirname "$(g++ -print-file-name=libstdc++.so.6 2>/dev/null)" 2>/dev/null || true)
  if [ -n "$GXX_LIB" ] && [ -f "$GXX_LIB/libstdc++.so.6" ] && grep -aq "GLIBCXX_3.4.32" "$GXX_LIB/libstdc++.so.6"; then
    export LD_LIBRARY_PATH="$GXX_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    log "using libstdc++ from $GXX_LIB"
  fi
  rm -rf "$HOME/.cache/flashinfer"/*/*/cached_ops/sampling 2>/dev/null || true   # drop the broken build
  log "starting Qwen ($OPEN_LLM) from $MPATH  (log: $VLOG)"
  HF_HUB_OFFLINE=1 "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server --model "$MPATH" \
      --served-model-name "$OPEN_LLM" --port "$QWEN_PORT" --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" \
      --max-model-len 32768 --gpu-memory-utilization ${QWEN_GPU_MEM:-0.85} > "$VLOG" 2>&1 &
  QWEN_PID=$!
  trap 'kill $QWEN_PID 2>/dev/null' EXIT
  local waited=0
  until curl -s "$QWEN_URL/models" | grep -q "\"$OPEN_LLM\""; do   # wait for OUR model on OUR port
    if ! kill -0 $QWEN_PID 2>/dev/null; then echo "ERROR: Qwen server exited - see $VLOG" >&2; tail -20 "$VLOG" >&2; exit 1; fi
    if [ $waited -ge 900 ]; then echo "ERROR: Qwen not ready after 15 min - see $VLOG" >&2; exit 1; fi
    sleep 10; waited=$((waited + 10))
  done
  log "Qwen ready after ${waited}s"
}

paraphrase() {   # adds free-text wordings (written by Qwen) + your hand-written prompts (bench/handwritten.csv)
  local UF=(); [ -f "$CTC_OUT/bench/handwritten.csv" ] && UF=(--user-file "$CTC_OUT/bench/handwritten.csv")
  OPENAI_API_KEY=EMPTY $PY paraphrase_bench --planner openai --model "$OPEN_LLM" --base-url "$QWEN_URL" "${UF[@]}"
}

qwen_bench() {   # GPU: Qwen answers every benchmark question (resumes; finished questions are kept)
  start_qwen
  paraphrase
  local LIM=(); [ -n "${QWEN_LIMIT:-}" ] && LIM=(--limit "$QWEN_LIMIT" --overwrite)
  for be in "${BACKENDS[@]}"; do
    if [ -d "$CTC_OUT/seg/$be" ]; then
      OPENAI_API_KEY=EMPTY $PY run_bench --backend "$be" --planner openai --model "$OPEN_LLM" \
          --base-url "$QWEN_URL" "${LIM[@]}"
    fi
  done
}

# ============================ pipelines ============================
gpu() {     # full run (trained models are skipped/restored if they already exist)
  prepare
  segment_oracle
  train_generalist
  segment_generalist
  eval_seg
  train_cls
  eval_cls
  make_bench
  qwen_bench
  report
}

resume() {  # after segmentation exists: evaluation, classifier, Qwen benchmark, figures
  eval_seg
  train_cls
  eval_cls
  make_bench
  qwen_bench
  report
}

refresh() { eval_seg; report; }    # re-score + re-draw only

report()  { $PY report; $PY figures --n-examples 3 --dpi 200; backup_results; }   # tables + PNGs + snapshot
figures() { $PY figures --n-examples 3 --dpi 200; }

# ---------------------------------------------------------------- dispatch
STAGE=${1:-gpu}
if [ "$STAGE" != "setup" ]; then activate; fi
cd "$CODE_DIR"
log "stage=$STAGE  data=$CTC_DATA  out=$CTC_OUT  host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || log "no GPU visible"
# GPU stages must never fall back to CPU silently (days instead of hours)
case "$STAGE" in
  gpu|resume|refresh|qwen_bench|train_generalist|segment_generalist|train_cls|selftest)
    if ! "$VENV/bin/python" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
      echo "ERROR: no working CUDA GPU on $(hostname) - this node is faulty." >&2
      echo "       Resubmit excluding it:  sbatch --exclude=$(hostname -s),g050,g051 run_hpc.sh $STAGE" >&2
      echo "       and report the node to it@gvsu.edu" >&2
      exit 2
    fi
    "$VENV/bin/python" -c "import torch; print('GPU OK:', torch.cuda.get_device_name(0))";;
esac
declare -F "$STAGE" >/dev/null || { echo "unknown stage: $STAGE"; exit 1; }
"$STAGE"
log "done: $STAGE"