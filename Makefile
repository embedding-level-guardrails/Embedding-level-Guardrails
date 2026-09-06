CONFIG ?= configs/rq1.yaml
PAIRS_CONFIG ?= configs/rq2_pairs.yaml
RQ3_CONFIG ?= configs/rq3_contrastive.yaml
OOD_CONFIG ?= configs/ood_toxicchat.yaml

.PHONY: setup data ood-data embed rq1 pairs rq3 lodo rq3-ablation mlflow test smoke smoke-rq3 clean

setup:
	pip install -r requirements.txt

data:
	python scripts/00_prepare_data.py --config $(CONFIG)

# OOD-набор для LODO: обучаемся на AEGIS, тестируем на реальном трафике ToxicChat
ood-data:
	python scripts/00_prepare_data.py --config $(OOD_CONFIG)

embed:
	python scripts/01_embed.py --config $(CONFIG)

rq1:
	bash scripts/run_rq1.sh

pairs:
	python scripts/07_build_pairs.py --config $(PAIRS_CONFIG)

# RQ3: три запуска с одними и теми же данными и разным objective — это и есть
# контролируемое сравнение contrastive против classification head.
rq3:
	python scripts/08_train_contrastive.py --config $(RQ3_CONFIG) --objective classification --run-name rq3-classification
	python scripts/08_train_contrastive.py --config $(RQ3_CONFIG) --objective contrastive    --run-name rq3-contrastive
	python scripts/08_train_contrastive.py --config $(RQ3_CONFIG) --objective joint          --run-name rq3-joint
	$(MAKE) lodo

# Сравнение на hold-out (AEGIS test) и OOD (ToxicChat) при пороге с AEGIS val.
lodo:
	python scripts/09_eval_lodo.py --config $(RQ3_CONFIG) --ood-config $(OOD_CONFIG)

# Абляция RQ2: вклад каждого типа пар при прочих равных.
rq3-ablation:
	python scripts/08_train_contrastive.py --config $(RQ3_CONFIG) --pair-types safe_harm_contrast --run-name abl-contrast-only
	python scripts/08_train_contrastive.py --config $(RQ3_CONFIG) --pair-types safe_harm_contrast paraphrase --run-name abl-plus-paraphrase
	python scripts/08_train_contrastive.py --config $(RQ3_CONFIG) --pair-types safe_harm_contrast jailbreak_variant --run-name abl-plus-jailbreak
	python scripts/08_train_contrastive.py --config $(RQ3_CONFIG) --pair-types safe_harm_contrast benign_twin --run-name abl-plus-twin

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

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

# офлайн-проверка RQ2/RQ3: маленькие пары + пара шагов обучения, без mlflow
smoke-rq3:
	python scripts/07_build_pairs.py --config $(PAIRS_CONFIG) --scale 0.02
	python scripts/08_train_contrastive.py --config $(RQ3_CONFIG) \
		--limit-pairs 64 --epochs 1 --batch-size 8 --eval-every 4 --no-mlflow \
		--run-name smoke
	python scripts/09_eval_lodo.py --config $(RQ3_CONFIG) --ood-config $(OOD_CONFIG) \
		--runs smoke --skip-base --n-boot 0

clean:
	rm -rf data/processed artifacts results mlruns mlartifacts mlflow.db
