.DEFAULT_GOAL := help

TRAIN_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ARGS ?=
backbone_name ?=

.PHONY: help linear-probe ettin-classifier embedder finetuned-ettin-classifier

help:
	@echo "Training targets: linear-probe, classifier, embedder, joint-classifier"

linear-probe:
	@cd "$(TRAIN_DIR)" && uv run -m ettin_guardrails.train training.models=linear-probe \
	training.epochs=3 \
	optimizer.lr=0.001 \
	+tokenizer.stratum_col=label \
	+training.validation.stratum_col=label \
	training.sampler.batch_size=64 \
	training.sampler.number_of_samples_per_category=32 \
	training.sampler.number_of_samples_per_epoch=45_000 \
	$(ARGS)

ettin-classifier:
	@cd "$(TRAIN_DIR)" && uv run -m ettin_guardrails.train training.models=classifier \
	training.epochs=7 \
	optimizer.lr=1e-6 \
	+tokenizer.stratum_col=label \
	+training.validation.stratum_col=label \
	training.sampler.batch_size=32 \
	training.sampler.number_of_samples_per_category=16 \
	training.sampler.number_of_samples_per_epoch=45_000 \
	$(ARGS)

embedder:
	@cd "$(TRAIN_DIR)" && uv run -m ettin_guardrails.train training.models=embedder \
	training.epochs=5 \
	optimizer.lr=1e-6 \
	training.sampler.batch_size=32 \
	training.sampler.number_of_samples_per_category=4 \
	training.sampler.number_of_samples_per_epoch=70_000 \
	$(ARGS)

finetuned-ettin-classifier:
	@cd "$(TRAIN_DIR)" && uv run -m ettin_guardrails.train training.models=classifier \
	training.epochs=7 \
	optimizer.lr=1e-6 \
	+models.classifier.backbone_name=$(backbone_name) \
	+tokenizer.stratum_col=label \
	+training.validation.stratum_col=label \
	training.sampler.batch_size=32 \
	training.sampler.number_of_samples_per_category=16 \
	training.sampler.number_of_samples_per_epoch=45_000 \
	scheduler.enabled=true \
    scheduler.start_factor=1.0 \
    scheduler.end_factor=0.1 \
    scheduler.total_iters=8 \
	$(ARGS)
