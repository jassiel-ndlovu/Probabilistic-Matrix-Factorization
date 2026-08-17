.PHONY: help install data test lint experiments figures tables report site notebook all clean

PY ?= python

help:
	@echo "install      install the package and dev extras"
	@echo "data         download MovieLens 100K and 1M"
	@echo "test         run the test suite"
	@echo "experiments  run the full study -> results/*.json"
	@echo "figures      render figures from results -> report/figures/"
	@echo "tables       generate the paper's numbers -> report/generated/"
	@echo "report       compile report/main.pdf"
	@echo "readme       refresh the README's results block"
	@echo "site         build the self-contained site/index.html"
	@echo "notebook     execute the walkthrough notebook in place"
	@echo "all          data -> test -> experiments -> figures -> tables -> report -> site"

install:
	$(PY) -m pip install -e ".[all]"

data:
	$(PY) scripts/download_data.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts

# The scalability experiment measures wall-clock time per epoch, so it wants a
# machine that is not otherwise busy. Avoid running this alongside the tuner.
experiments:
	$(PY) scripts/run_experiments.py

quick:
	$(PY) scripts/run_experiments.py --quick

legacy:
	$(PY) scripts/reproduce_legacy.py --seeds 10

tune:
	$(PY) scripts/tune.py --dataset ml-100k
	$(PY) scripts/tune.py --dataset ml-1m

figures:
	$(PY) scripts/make_figures.py

tables:
	$(PY) scripts/make_report_tables.py

readme:
	$(PY) scripts/update_readme.py

report: figures tables
	cd report && latexmk -pdf -silent main.tex

site:
	$(PY) scripts/build_site.py

notebook:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=1800 notebooks/pmf_walkthrough.ipynb

all: data test experiments figures tables report readme site

clean:
	cd report && latexmk -C >/dev/null 2>&1 || true
	rm -rf report/generated report/figures/*.pdf report/figures/*.png
	rm -rf .pytest_cache **/__pycache__
