#!/usr/bin/env bash
# One-command setup: Python environment, models, data, checks.
#
#   ./setup.sh                       # everything, with a ~78k-document corpus (~45 min to embed on CPU)
#   ./setup.sh --sample 5000         # smaller corpus, ready in minutes (all 722 gold docs are always included)
#   ./setup.sh --skip-data           # environment and models only
#   ./setup.sh --all-models          # also pull the ablation candidates and the judge
#   VENV=../venv ./setup.sh          # use an existing virtualenv
#
# Every step is safe to re-run: installs are idempotent, the data load only adds what
# is missing, and nothing existing is deleted. See docs/WORKFLOW.md for what each step does.
set -euo pipefail

cd "$(dirname "$0")"

VENV="${VENV:-.venv}"
SAMPLE=78053
SKIP_DATA=0
SKIP_MODELS=0
SKIP_TESTS=0
ALL_MODELS=0
GENERATOR_MODEL="${GENERATOR_MODEL:-qwen2.5:7b-instruct-q4_K_M}"
ABLATION_MODELS=(phi3:mini-4k phi:2.7b llama3.1:8b-instruct-q4_K_M)

usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --sample) SAMPLE="$2"; shift ;;
    --skip-data) SKIP_DATA=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    --all-models) ALL_MODELS=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32mok\033[0m  %s\n' "$*"; }
warn() { printf '    \033[33m!!\033[0m  %s\n' "$*"; }
die()  { printf '    \033[31mxx\033[0m  %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
step "Python"
PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null || die "python3 not found; install Python 3.10 or newer"
"$PYTHON_BIN" -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
  || die "Python 3.10+ required, found $("$PYTHON_BIN" -V 2>&1)"
ok "$("$PYTHON_BIN" -V 2>&1)"

step "Virtual environment ($VENV)"
if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$VENV" || die "could not create $VENV (on Debian/Ubuntu: sudo apt install python3-venv)"
  ok "created"
else
  ok "reusing existing environment"
fi
PY="$VENV/bin/python"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -r requirements.txt
ok "requirements installed"
export PYTHONPATH=".:scripts"

# ---------------------------------------------------------------------------
step "Ollama (local LLM server)"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
if [ "$SKIP_MODELS" = 1 ]; then
  ok "skipped (--skip-models)"
elif ! command -v ollama >/dev/null; then
  warn "Ollama is not installed - answers cannot be generated until it is."
  warn "Install it from https://ollama.com/download, then re-run ./setup.sh"
elif ! curl -fsS -m 3 "$OLLAMA_HOST/api/version" >/dev/null; then
  warn "Ollama is installed but not running at $OLLAMA_HOST (start it with: ollama serve)"
else
  models=("$GENERATOR_MODEL")
  [ "$ALL_MODELS" = 1 ] && models+=("${ABLATION_MODELS[@]}")
  for model in "${models[@]}"; do
    if ollama list | awk 'NR>1 {print $1}' | grep -qx "$model"; then
      ok "$model already present"
    else
      echo "    pulling $model ..."
      ollama pull "$model" && ok "$model pulled"
    fi
  done
fi

# ---------------------------------------------------------------------------
step "Data (EnterpriseRAG-Bench in ChromaDB)"
CHROMA_PATH_RESOLVED="$("$PY" -c 'import config; print(config.CHROMA_PATH)')"
echo "    Chroma store: $CHROMA_PATH_RESOLVED  (override with CHROMA_PATH=...)"
if [ "$SKIP_DATA" = 1 ]; then
  ok "skipped (--skip-data)"
else
  docs_count="$("$PY" - <<'EOF' 2>/dev/null || echo 0
import chromadb, config
try:
    print(chromadb.PersistentClient(path=config.CHROMA_PATH).get_collection(config.CHROMA_DOCS_COLLECTION).count())
except Exception:
    print(0)
EOF
)"
  if [ "${docs_count:-0}" -gt 0 ]; then
    ok "docs collection already has $docs_count documents - topping up questions only"
    "$PY" scripts/load_dataset.py --questions-only
  else
    echo "    downloading the benchmark and embedding a $SAMPLE-document corpus (resumable; re-run if interrupted)"
    "$PY" scripts/load_dataset.py --sample "$SAMPLE"
  fi
  "$PY" scripts/add_gold_docs.py
  ok "every gold document is in the corpus"
fi

# ---------------------------------------------------------------------------
step "Service check (builds the BM25 cache on first run)"
if [ "$SKIP_DATA" = 1 ] && [ "${docs_count:-0}" = 0 ] && ! "$PY" -c 'import chromadb, config; chromadb.PersistentClient(path=config.CHROMA_PATH).get_collection(config.CHROMA_DOCS_COLLECTION)' 2>/dev/null; then
  warn "no corpus yet - skipped (run ./setup.sh without --skip-data)"
else
  "$PY" - <<'EOF'
from core.service import RAGService
service = RAGService()
service.initialize()
status = service.status()
print(f"    ready={status['ready']} bm25={status['bm25_built']} generator={status['generator_loaded']} "
      f"calibrator={status['calibrator_fitted']} ({status['calibrator_examples']} examples)")
if status["error"]:
    print("    note:", status["error"])
EOF
  ok "service initialises"
fi

# ---------------------------------------------------------------------------
step "Tests"
if [ "$SKIP_TESTS" = 1 ]; then
  ok "skipped (--skip-tests)"
else
  "$PY" -m unittest discover -s tests -t . 2>&1 | tail -3
fi

# ---------------------------------------------------------------------------
step "Done"
cat <<EOF
    Start the website and API:
        $VENV/bin/uvicorn main:app --port 8000

    Then open:
        http://localhost:8000/            Ask - real-time RAG engine
        http://localhost:8000/dashboard   Monitor - experiments and results
        http://localhost:8000/docs        API reference

    Re-run every evaluation (results are saved to results/):
        scripts/run_pipeline.sh
EOF
