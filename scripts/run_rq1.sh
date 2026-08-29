#!/usr/bin/env bash
# Полный прогон RQ1. Использование:
#   bash scripts/run_rq1.sh                    # с конфигом по умолчанию
#   CONFIG=configs/rq1_defensive.yaml bash scripts/run_rq1.sh
set -euo pipefail

CONFIG="${CONFIG:-configs/rq1.yaml}"

echo "== 0. Подготовка данных =="
python scripts/00_prepare_data.py --config "$CONFIG"

echo "== 1. Эмбеддинги =="
python scripts/01_embed.py --config "$CONFIG"

echo "== 2. Косинусы и геометрия =="
python scripts/02_rq1_similarity.py --config "$CONFIG"

echo "== 3. Пробы =="
python scripts/03_rq1_probe.py --config "$CONFIG"

echo "== 4. Проекции =="
python scripts/04_rq1_viz.py --config "$CONFIG"

echo "== 5. Error analysis =="
python scripts/06_error_analysis.py --config "$CONFIG"

echo "== 6. Отчёт =="
python scripts/05_report.py --config "$CONFIG"

echo "Готово: results/rq1/REPORT.md"
