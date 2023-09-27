# Makefile

# OS specific commands and variables
ifeq ($(OS),Windows_NT)
    SET_ENV = set
else
    SET_ENV = export
endif

# List of directories and files to format and lint
TARGETS = main.py

# Format code using isort and black
format:
	poetry run isort $(TARGETS)
	poetry run black $(TARGETS)

# Lint code using ruff
lint:
	poetry run ruff $(TARGETS)

# Setup project for development
setup:
	poetry install

# Run the app
run:
	streamlit run main.py

# Display help message by default
.DEFAULT_GOAL := help
help:
	@echo "Available commands:"
	@echo "  make format      - Format code using isort and black"
	@echo "  make lint        - Lint code using ruff"
	@echo "  make check       - Format and lint code"
	@echo "  make setup       - Setup project for development"
	@echo "  make run         - Run the app"

# Declare the targets as phony
.PHONY: format lint check help
