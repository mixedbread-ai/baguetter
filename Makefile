.PHONY: help install dev-install format lint test clean update-deps

GIT_ROOT ?= $(shell git rev-parse --show-toplevel)
PYTHON_VERSION ?= 3.11
UV_VERSION ?= 0.1.0

help:	## Show all Makefile targets.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

install:	## Install the package and dependencies.
	uv venv
	uv pip install -e .

dev-install:	## Install the package, dependencies, and development tools.
	uv venv
	uv pip install -e .[dev]

format:	## Run code autoformatters (ruff).
	uv run ruff format .

lint:	## Run linters: ruff
	uv run ruff check .

test:	## Run tests via pytest
	uv run pytest

clean:	## Clean up build artifacts.
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '*.pyc' -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf dist
	rm -rf build
	rm -rf .venv

update-deps:	## Update dependencies.
	uv pip compile pyproject.toml -o requirements.txt
	uv pip sync requirements.txt
	uv run pre-commit autoupdate

setup-uv:	## Setup uv with the specified version
	curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --version $(UV_VERSION)