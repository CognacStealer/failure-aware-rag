#!/usr/bin/env bash
# Run the full evaluation workflow, one step at a time (the laptop is CPU-only, so
# steps never compete for cores). Every step saves into results/ and is safe to
# re-run: the calibrator evaluation and the generator ablation resume where they stopped.
#
#   scripts/run_pipeline.sh            # all steps
#   scripts/run_pipeline.sh ablation   # only the named steps, in the order given
#
# Logs: data/logs/<step>.log
set -uo pipefail

cd "$(dirname "$0")/.."
# The virtualenv setup.sh creates (.venv), else the author's sibling ../venv.
if [ -z "${PYTHON:-}" ]; then [ -x .venv/bin/python ] && PYTHON=.venv/bin/python || PYTHON=../venv/bin/python; fi
export PYTHONPATH=".:scripts"
# Leave cores free for the interactive website: 6 of 12 threads for torch/BLAS here,
# and 6 Ollama threads for the ablation's model calls.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"
mkdir -p data/logs

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(train_calibrator evaluate_retrieval evaluate_calibrator ablation)

run_step() {
  local name=$1; shift
  echo "[$(date '+%F %T')] start $name" | tee -a data/logs/pipeline.log
  if "$PYTHON" "$@" >"data/logs/$name.log" 2>&1; then
    echo "[$(date '+%F %T')] done  $name" | tee -a data/logs/pipeline.log
  else
    echo "[$(date '+%F %T')] FAIL  $name (see data/logs/$name.log)" | tee -a data/logs/pipeline.log
  fi
}

for step in "${steps[@]}"; do
  case "$step" in
    train_calibrator)    run_step "$step" scripts/train_calibrator.py ;;
    evaluate_retrieval)  run_step "$step" scripts/evaluate_retrieval.py --split test ;;
    evaluate_calibrator) run_step "$step" scripts/evaluate_calibrator.py ;;
    ablation)            run_step "$step" scripts/run_local_generator_ablation.py --threads 6 ;;
    *) echo "unknown step: $step" >&2; exit 2 ;;
  esac
done
