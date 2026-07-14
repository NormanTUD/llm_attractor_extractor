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
GROUPS="${GROUPS:-}"
DIMS="${DIMS:-}"

# --- Ensure uv is available ---
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo "uv version: $(uv --version)"
echo "Groups: ${GROUPS:-all}"
echo "Dims: ${DIMS:-auto}"
echo ""

# --- Verify script and data exist ---
if [ ! -f "residual_attractors.py" ]; then
    echo "ERROR: residual_attractors.py not found in $(pwd)"
    ls -la *.py 2>/dev/null || echo "  (none)"
    exit 1
fi

# --- Build command ---
CMD=(uv run residual_attractors.py)

if [ -n "$GROUPS" ]; then
    CMD+=(--groups $GROUPS)
fi

if [ -n "$DIMS" ]; then
    CMD+=(--dims "$DIMS")
fi

echo "Running: ${CMD[*]}"
echo "============================================"
echo ""

"${CMD[@]}"

echo ""
echo "============================================"
echo "Job finished: $(date)"
echo "============================================"
