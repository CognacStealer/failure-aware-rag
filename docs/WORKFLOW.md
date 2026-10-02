# Workflow

From an empty machine to a running website and saved results. Commands run from the
repository root; `./setup.sh` creates the virtualenv at `.venv`.

```mermaid
flowchart TD
    S1[1. Install] --> S2[2. Load data into Chroma]
    S2 --> S3[3. Add missing gold documents]
    S3 --> S4[4. Start the API<br/>builds BM25 cache]
    S4 --> S5[5. Train calibrator]
    S5 --> S6[6. Evaluate retrieval]
    S6 --> S7[7. Evaluate calibrator]
    S7 --> S8[8. Generator ablation]
    S5 & S6 & S7 & S8 --> R[(results/)]
    S4 --> W[Website: / and /dashboard]
```

## 1-3. Setup in one command

```bash
./setup.sh
```

It checks Python (3.10+), creates `.venv` and installs `requirements.txt`, checks
Ollama and pulls the generator, loads the dataset into ChromaDB, adds every gold
document, initialises the service (building the BM25 cache) and runs the tests. Every
step is safe to re-run.

| Option | Effect |
|---|---|
| `--sample N` | corpus size (default 78,053; all 722 gold documents are always included) |
| `--skip-data` | environment and models only |
| `--all-models` | also pull the ablation candidates (phi3, phi-2) and the judge (Llama 3.1 8B) |
| `--skip-models`, `--skip-tests` | skip those steps |
| `VENV=path` | use an existing virtualenv; `CHROMA_PATH=path` chooses the Chroma store |

The steps it runs, if you prefer to do them by hand:

1. **Install:** `python -m venv .venv && .venv/bin/pip install -r requirements.txt`, then install
   [Ollama](https://ollama.com/download) and `ollama pull qwen2.5:7b-instruct-q4_K_M`.
2. **Load data:** `PYTHONPATH=. .venv/bin/python scripts/load_dataset.py` downloads EnterpriseRAG-Bench
   and creates the `Questions` collection (gold document IDs, gold answer, answer facts and question
   type as metadata) and the `docs` corpus (every gold document plus a seeded random sample, embedded
   with all-MiniLM-L6-v2). Resumable and additive. See [DATASET.md](DATASET.md).
3. **Gold documents:** `PYTHONPATH=. .venv/bin/python scripts/add_gold_docs.py` adds any gold
   document missing from an existing corpus (a no-op after step 2).

## 4. Start the API and website

```bash
.venv/bin/uvicorn main:app --port 8000
```

- `http://localhost:8000/` - **Ask**: the real-time engine
- `http://localhost:8000/dashboard` - **Monitor**: experiments and results
- `http://localhost:8000/docs` - OpenAPI reference

The first start builds the BM25 cache (~1 minute); later starts take ~1.5 s.

## 5-8. Experiments

One command runs every step in order, each saving to `results/`:

```bash
scripts/run_pipeline.sh                      # all steps
scripts/run_pipeline.sh evaluate_calibrator  # or named steps only
```

| Step | Script | Output | Time (CPU laptop) |
|---|---|---|---|
| Train calibrator | `scripts/train_calibrator.py` | `models/calibrator.pkl`, `results/calibrator_training/` | ~2 min |
| Evaluate retrieval | `scripts/evaluate_retrieval.py` | `results/retrieval_eval/` | ~15 min |
| Evaluate calibrator | `scripts/evaluate_calibrator.py` | `results/calibrator_eval/` | ~1 h |
| Generator ablation | `scripts/run_local_generator_ablation.py` | `data/ablation_runs/full_benchmark/`, snapshots in `results/generator_ablation/` | ~2 days for 500 questions |

The calibrator evaluation and the ablation are **crash-safe**: they record each question
as it finishes and resume after an interruption. Logs are in `data/logs/`.

Snapshot an in-progress ablation at any time:

```bash
PYTHONPATH=.:scripts .venv/bin/python scripts/run_local_generator_ablation.py --summarize --save
```

## Results store

Every result is a folder `results/<kind>/<timestamp>/` with `result.json` (metrics),
per-question CSVs and `meta.json` (git commit, full config snapshot, command, Python
version). `results/index.json` and `results/README.md` list them all; the Monitor's
**Results** view browses and downloads them.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -t .
```

Unit tests use fakes, so they need neither Chroma nor Ollama.

## Laptop notes

The reference machine is a CPU-only i5-1335U with 14 GB RAM. Steps run one at a time;
the pipeline caps itself at 6 threads so the website stays responsive; Ollama unloads
candidate models after each call because its prompt cache otherwise grows until the
kernel kills it.
