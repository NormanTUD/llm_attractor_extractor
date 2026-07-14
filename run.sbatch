#!/bin/bash
#SBATCH -N1
#SBATCH --job-name=residual-attractors
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6

# ============================================================================
# Residual Stream Attractor Extraction — SLURM Batch Job
#
# Saves RAW residual streams (all layers, all sequence positions) as .npy
# plus the final-token streams and attractor metrics.
#
# Output goes to $OUTPUT_DIR (default: attractor_data_raw_$SLURM_JOB_ID)
# ============================================================================

set -euo pipefail

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "GPUs:          $CUDA_VISIBLE_DEVICES"
echo "Start time:    $(date)"
echo "Working dir:   $(pwd)"
echo "============================================"

# --- Configuration ---
MODEL="${MODEL:-deepseek-ai/deepseek-llm-7b-base}"
DTYPE="${DTYPE:-float16}"
OUTPUT_DIR="${OUTPUT_DIR:-attractor_data_raw_${SLURM_JOB_ID}}"
GROUPS="${GROUPS:-}"  # empty = all groups

# --- Ensure uv is available ---
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo "uv version: $(uv --version)"
echo "Model: $MODEL"
echo "Dtype: $DTYPE"
echo "Output: $OUTPUT_DIR"
echo ""

# --- Build command ---
CMD="uv run residual_attractors.py \
    --model $MODEL \
    --device cuda \
    --dtype $DTYPE \
    --output $OUTPUT_DIR \
    --save-full-streams"

if [ -n "$GROUPS" ]; then
    CMD="$CMD --groups $GROUPS"
fi

echo "Running: $CMD"
echo "============================================"
echo ""

eval $CMD

echo ""
echo "============================================"
echo "Job finished:  $(date)"
echo "Output in:     $OUTPUT_DIR"
echo "Raw streams:   $(find $OUTPUT_DIR -name '*_full_streams.npy' | wc -l) files"
echo "Total size:    $(du -sh $OUTPUT_DIR | cut -f1)"
echo "============================================"
