.PHONY: install lint format test run typecheck

install:
	pip install -r requirements.txt

typecheck:
	python -m mypy detection/ streaming/ scripts/ --ignore-missing-imports --strict-optional

lint: typecheck
	ruff check .
	black --check .

format:
	ruff check --fix .
	black .

test:
	pytest -q

run:
	python run_pipeline.py
