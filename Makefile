.PHONY: install lint format test run scale-workers

install:
	pip install -r requirements.txt

lint:
	ruff check .
	black --check .

format:
	ruff check --fix .
	black .

test:
	pytest -q

run:
	python run_pipeline.py

scale-workers:
	@if [ -z "$(N)" ]; then \
		echo "Error: N is required. Usage: make scale-workers N=4"; \
		exit 1; \
	fi
	python -m scripts.kafka_workers --num-workers $(N)
