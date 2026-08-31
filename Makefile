.PHONY: install test serve bench inspect

install:
	pip install -r requirements.txt

test:
	pytest

serve:
	python -m server.main

bench:
	python bench.py

inspect:
	python scratch/inspect_model.py
