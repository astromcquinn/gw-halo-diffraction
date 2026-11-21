.PHONY: all figures test note
all: test figures note
test:
	python -m unittest discover -s tests -v
figures:
	OPENBLAS_NUM_THREADS=1 python scripts/run_calculations.py
	python scripts/plot_diagnostics.py
note:
	cd docs && latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
