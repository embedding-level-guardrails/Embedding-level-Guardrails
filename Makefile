CONFIG ?= configs/rq1.yaml

.PHONY: setup data embed rq1 test smoke clean

setup:
	pip install -r requirements.txt

data:
	python scripts/00_prepare_data.py --config $(CONFIG)

embed:
	python scripts/01_embed.py --config $(CONFIG)

rq1:
	bash scripts/run_rq1.sh

test:
	pytest -q tests/

# офлайн-проверка пайплайна: синтетика + хеш-энкодер, без сети и torch
smoke:
	python scripts/00_prepare_data.py --config $(CONFIG) --synthetic 600
	python scripts/01_embed.py --config $(CONFIG) --encoders dummy
	python scripts/02_rq1_similarity.py --config $(CONFIG)
	python scripts/03_rq1_probe.py --config $(CONFIG)
	python scripts/04_rq1_viz.py --config $(CONFIG) --methods pca
	python scripts/06_error_analysis.py --config $(CONFIG)
	python scripts/05_report.py --config $(CONFIG)

clean:
	rm -rf data/processed artifacts results
