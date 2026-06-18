SHELL := /bin/bash
.DEFAULT_GOAL := help

UV ?= uv
CONFIG ?=
CONFIG_FLAG := $(if $(CONFIG),--config $(CONFIG),)
ARTIFACT_DIR ?= data/demo

.PHONY: help sync demo run serve quick-demo build test lint format format-check typecheck check clean clean-artifacts clean-caches distclean

help: ## Show available targets.
	@awk 'BEGIN {FS = ":.*##"; printf "Peakle targets:\n"} /^[a-zA-Z0-9_-]+:.*##/ {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

sync: ## Install project and development dependencies with uv.
	$(UV) sync

demo: ## Generate the full synthetic demo artifacts.
	$(UV) run peakle $(CONFIG_FLAG) demo run

run: demo serve ## Generate artifacts, then serve the browser viewer.

serve: ## Serve generated artifacts in the browser.
	$(UV) run peakle $(CONFIG_FLAG) web serve

quick-demo: ## Generate a smaller, faster demo for smoke testing.
	$(UV) run peakle $(CONFIG_FLAG) demo run \
		--output $(ARTIFACT_DIR) \
		--grid-width 96 \
		--grid-height 72 \
		--image-width 480 \
		--image-height 270 \
		--optimization-max-iterations 40

build: ## Build source and wheel distributions with uv_build.
	$(UV) build

test: ## Run tests.
	$(UV) run pytest

lint: ## Run Ruff lint checks.
	$(UV) run ruff check .

format: ## Format Python files with Ruff.
	$(UV) run ruff format .

format-check: ## Check Python formatting without modifying files.
	$(UV) run ruff format . --check

typecheck: ## Run ty type checks.
	$(UV) run ty check

check: format-check lint test typecheck ## Run all verification checks.

clean: clean-artifacts clean-caches ## Remove generated artifacts and tool caches.
	rm -rf dist

clean-artifacts: ## Remove generated demo artifacts.
	rm -rf $(ARTIFACT_DIR)

clean-caches: ## Remove local test/lint/typecheck caches.
	rm -rf .pytest_cache .ruff_cache .ty
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

distclean: clean ## Remove generated artifacts, caches, and the uv virtualenv.
	rm -rf .venv
