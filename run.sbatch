#!/bin/bash
#SBATCH -N1
#SBATCH --job-name=residual-attractors
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6

set -euxo pipefail

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "GPUs:          ${CUDA_VISIBLE_DEVICES:-none}"
echo "Start time:    $(date)"
echo "Working dir:   $(pwd)"
echo "============================================"

# --- Configuration ---
MODEL="${MODEL:-deepseek-ai/deepseek-llm-7b-base}"
DTYPE="${DTYPE:-float16}"
OUTPUT_DIR="${OUTPUT_DIR:-attractor_data_raw_${SLURM_JOB_ID}}"
GROUPS="${GROUPS:-200270}"

# --- Ensure uv is available ---
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo "uv version: $(uv --version)"
echo "Model: $MODEL"
echo "Dtype: $DTYPE"
echo "Output: $OUTPUT_DIR"
echo ""

# --- Run extraction ---
# NOTE: residual_attractors.py muss --model, --device, --dtype, --output,
#       --save-full-streams akzeptieren! Aktuell ist es nur ein Viewer.
#       Falls du ein separates extract-Skript hast, ändere SCRIPT= hier:
SCRIPT="residual_attractors.py"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: $SCRIPT not found!"
    ls -la *.py 2>/dev/null
    exit 1
fi

CMD=(
    uv run "$SCRIPT"
    --model "$MODEL"
    --device cuda
    --dtype "$DTYPE"
    --output "$OUTPUT_DIR"
    --save-full-streams
)

if [ -n "$GROUPS" ]; then
    CMD+=(--groups $GROUPS)
fi

echo "Running: ${CMD[*]}"
echo "============================================"

"${CMD[@]}"

echo ""
echo "============================================"
echo "Job finished:  $(date)"
echo "Output in:     $OUTPUT_DIR"
echo "Size:          $(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)"
echo "============================================"
