.PHONY: install test serve bench inspect

install:
	pip install -r requirements.txt

test:
	pytest

serve:
	uvicorn server.app:app --host 0.0.0.0 --port 8000

bench:
	python bench.py

inspect:
	python scratch/inspect_model.py
