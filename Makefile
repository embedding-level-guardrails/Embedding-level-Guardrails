CONFIG ?= configs/rq1.yaml
PAIRS_CONFIG ?= configs/rq2_pairs.yaml
RQ3_CONFIG ?= configs/rq3_contrastive.yaml
OOD_CONFIG ?= configs/ood_toxicchat.yaml
WGM_CONFIG ?= configs/rq2_pairs_wildguardmix.yaml
WGM_RQ3_CONFIG ?= configs/rq3_wildguardmix.yaml
PURE_PAIRS_CONFIG ?= configs/rq2_pairs_wgm_pure.yaml
PURE_CONFIG ?= configs/rq3_wgm_pure.yaml

.PHONY: setup data ood-data wgm-pairs rq3-wgm lodo-wgm geometry-wgm export-hf pure-data rq3-pure lodo-pure geometry-pure embed rq1 pairs rq3 lodo rq3-ablation mlflow test smoke smoke-rq3 clean

setup:
	pip install -r requirements.txt

data:
	python scripts/00_prepare_data.py --config $(CONFIG)

# OOD-набор для LODO: обучаемся на AEGIS, тестируем на реальном трафике ToxicChat
ood-data:
	python scripts/00_prepare_data.py --config $(OOD_CONFIG)

# WildGuardMix: данные -> эмбеддинги e5 для майнинга benign twins -> пары.
# Датасет gated: нужен принятый запрос доступа и HF_TOKEN.
wgm-pairs:
	python scripts/00_prepare_data.py --config $(WGM_CONFIG)
	python scripts/01_embed.py --config $(WGM_CONFIG)
	python scripts/07_build_pairs.py --config $(WGM_CONFIG)

# Собрать дообученную модель для Hugging Face Hub (без загрузки).
# RUN — каталог прогона, REPO — будущий репозиторий. Загрузка: scripts/12_push_to_hub.py
RUN ?= wgm-joint
REPO ?= USER/e5-small-v2-wildguardmix-joint
export-hf:
	python scripts/11_export_hf.py --config $(WGM_RQ3_CONFIG) --run $(RUN) --repo-id $(REPO)

# «Чистый» WildGuardMix: данные -> эмбеддинги e5 -> триплеты в режиме coverage.
# Если wgm-pairs уже запускался, шаги 00 и 01 просто переиспользуют готовое.
pure-data:
	python scripts/00_prepare_data.py --config $(PURE_PAIRS_CONFIG)
	python scripts/01_embed.py --config $(PURE_PAIRS_CONFIG)
	python scripts/07_build_pairs.py --config $(PURE_PAIRS_CONFIG)

# Пять вариантов обучения на одних и тех же текстах + оценка + геометрия.
# Результаты: results/wgm_pure/rq3/, чекпоинты: artifacts/runs_wgm_pure/
rq3-pure:
	python scripts/08_train_contrastive.py --config $(PURE_CONFIG) --objective classification --run-name pure-classification
	python scripts/08_train_contrastive.py --config $(PURE_CONFIG) --objective contrastive    --run-name pure-contrastive
	python scripts/08_train_contrastive.py --config $(PURE_CONFIG) --objective joint          --run-name pure-joint
	python scripts/08_train_contrastive.py --config $(PURE_CONFIG) --objective supcon         --run-name pure-supcon
	python scripts/08_train_contrastive.py --config $(PURE_CONFIG) --objective supcon_joint   --run-name pure-supcon-joint
	$(MAKE) lodo-pure
	$(MAKE) geometry-pure

lodo-pure:
	python scripts/09_eval_lodo.py --config $(PURE_CONFIG) --ood-config $(OOD_CONFIG)

geometry-pure:
	python scripts/10_embedding_geometry.py --config $(PURE_CONFIG) --ood-config $(OOD_CONFIG)

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

# RQ3 на WildGuardMix: три objective + LODO + геометрия. Результаты: results/wildguardmix/rq3/
rq3-wgm:
	python scripts/08_train_contrastive.py --config $(WGM_RQ3_CONFIG) --objective classification --run-name wgm-classification
	python scripts/08_train_contrastive.py --config $(WGM_RQ3_CONFIG) --objective contrastive    --run-name wgm-contrastive
	python scripts/08_train_contrastive.py --config $(WGM_RQ3_CONFIG) --objective joint          --run-name wgm-joint
	$(MAKE) lodo-wgm
	$(MAKE) geometry-wgm

lodo-wgm:
	python scripts/09_eval_lodo.py --config $(WGM_RQ3_CONFIG) --ood-config $(OOD_CONFIG)

# Как objective перестроил пространство: порядок в триплетах и геометрия.
# Результат: results/wildguardmix/rq3/GEOMETRY.md
geometry-wgm:
	python scripts/10_embedding_geometry.py --config $(WGM_RQ3_CONFIG) --ood-config $(OOD_CONFIG)

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
