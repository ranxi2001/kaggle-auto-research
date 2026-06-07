.PHONY: install dev test lint clean

install:
	pip install -e .

dev:
	pip install -e ".[dev,deep]"

test:
	pytest tests/ -v

lint:
	ruff check src/ cli/ tests/
	ruff format --check src/ cli/ tests/

format:
	ruff format src/ cli/ tests/

clean:
	rm -rf build/ dist/ *.egg-info __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
