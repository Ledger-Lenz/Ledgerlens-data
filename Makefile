.PHONY: install lint format test typecheck run

install:
	pip install -r requirements.txt

lint:
	ruff check .
	black --check .
	$(MAKE) typecheck

format:
	ruff check --fix .
	black .

test:
	pytest -q

typecheck:
	mypy detection streaming scripts

run:
	python run_pipeline.py
