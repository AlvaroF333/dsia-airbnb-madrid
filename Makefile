.PHONY: setup lint format test cov

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	pytest -q

cov:
	pytest -q --cov --cov-report=term-missing --cov-fail-under=70
