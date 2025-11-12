.PHONY: build test clean install help

help:
	@echo "Kelly Asset Pack Generator - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make build       Build all Kelly assets"
	@echo "  make test        Run tests"
	@echo "  make clean       Remove generated files"
	@echo "  make install     Install dependencies"
	@echo "  make help        Show this help"

install:
	pip install -r requirements.txt

build:
	python -m kelly_pack.cli build --outdir ./output

test:
	pytest tests/ -v

clean:
	rm -rf output/ weights/ __pycache__/ kelly_pack/__pycache__/ tests/__pycache__/ .pytest_cache/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete


