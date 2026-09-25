.DEFAULT_GOAL := help

TRAIN_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ARGS ?=
num_per_category ?=
batch_size ?=

.PHONY: help linear-probe classifier embedder joint-classifier

help:
	@echo "Training targets: linear-probe, classifier, embedder, joint-classifier"

linear-probe:
	@cd "$(TRAIN_DIR)" && uv run -m ettin_guardrails.train training.models=linear-probe \
	training.epochs=3 \
	optimizer.lr=0.001 \
	+tokenizer.stratum_col=label \
	training.sampler.batch_size=$(batch_size) \
	training.sampler.number_of_samples_per_category=$(num_per_category) \
	training.sampler.number_of_samples_per_epoch=45_000 \
	$(ARGS)

classifier:
	@cd "$(TRAIN_DIR)" && uv run -m ettin_guardrails.train training.models=classifier \
	training.epochs=5 \
	optimizer.lr=1e-6 \
	+tokenizer.stratum_col=label \
	training.sampler.batch_size=$(batch_size) \
	training.sampler.number_of_samples_per_category=$(num_per_category) \
	training.sampler.number_of_samples_per_epoch=45_000 \
	$(ARGS)

embedder:
	@cd "$(TRAIN_DIR)" && uv run -m ettin_guardrails.train training.models=embedder \
	training.epochs=4 \
	optimizer.lr=1e-6 \
	+tokenizer.stratum_col=label \
	training.sampler.batch_size=$(batch_size) \
	training.sampler.number_of_samples_per_category=$(num_per_category) \
	training.sampler.number_of_samples_per_epoch=70_000 \
	$(ARGS)

joint-classifier:
	@echo "Not yet implemented"
