# Shortcuts for the common commands. Each target calls the existing scripts;
# see setup.sh and scripts/run_pipeline.sh for what they do.
# The virtualenv setup.sh creates (.venv), else the sibling ../venv, as in run_pipeline.sh.
VENV ?= $(if $(wildcard .venv/bin/python),.venv,../venv)
PY := $(VENV)/bin/python
export PYTHONPATH := .:scripts

.PHONY: help setup calibrator eval ablation snapshot serve test

help:  ## list targets
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

setup:  ## environment, models, data, calibrator, checks (./setup.sh)
	./setup.sh

calibrator:  ## (re)train the routing calibrator, ~2 min on CPU
	scripts/run_pipeline.sh train_calibrator

eval:  ## retrieval and calibrator evaluations on the test split
	scripts/run_pipeline.sh evaluate_retrieval evaluate_calibrator

ablation:  ## generator comparison on all 500 questions (resumable, ~days on CPU)
	scripts/run_pipeline.sh ablation

snapshot:  ## save the in-progress ablation results to results/
	$(PY) scripts/run_local_generator_ablation.py --summarize --save

serve:  ## start the API and website on http://localhost:8000
	$(VENV)/bin/uvicorn main:app --port 8000

test:  ## unit tests (no Chroma or Ollama needed)
	$(PY) -m unittest discover -s tests -t .
