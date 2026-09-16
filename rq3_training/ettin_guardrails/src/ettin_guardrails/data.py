import random
from collections.abc import Hashable, Iterable, Iterator

import polars as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, Sampler
from transformers import AutoTokenizer

LABEL_MAPPING_DICT = {"unsafe": "harmful", "safe": "safe", "harm": "harmful"}

JBB_CATEGORY_MAPPING_DICT = {
    "chemical_biological": "Controlled/Regulated Substances",
    "cybercrime_intrusion": "Malware",
    "harmful": "Threat",
    "copyright": "Copyright/Trademark/Plagiarism",
    "misinformation_disinformation": "Fraud/Deception",
    "harassment_bullying": "Harassment",
    "illegal": "Criminal Planning/Confessions"
}


def load_data(parameters: DictConfig) -> pl.DataFrame:
    prompt_type_col = "prompt_type"
    label_type_col = "label_type"
    only_matching_rows_expr = (
            (pl.col(label_type_col) == "anchor_label") & (pl.col(prompt_type_col) == "anchor_text") |
            (pl.col(label_type_col) == "pair_label") & (pl.col(prompt_type_col) == "pair_text")
    )
    keep_only_first_cat = (
        pl.col("category")
        .str.split(",")
        .list.first()
    )
    dataset = (
        pl.read_ndjson(parameters.data.path)
        .unpivot(
            on=["anchor_text", "pair_text"],
            index=["category", "pair_type", "anchor_label", "pair_label"],
            variable_name=prompt_type_col,
            value_name="prompt",
        ).unpivot(
            on=["anchor_label", "pair_label"],
            index=["category", "pair_type", prompt_type_col, "prompt"],
            variable_name=label_type_col,
            value_name="label",
        ).filter(only_matching_rows_expr)
        .drop(label_type_col, prompt_type_col, "pair_type")
        .with_columns(keep_only_first_cat.alias("category"))
        .with_columns(
            pl.col("label").replace_strict(LABEL_MAPPING_DICT).alias("label"),
        )
        .with_columns(
            pl.when(pl.col("label") == "safe")
            .then(pl.lit(""))
            .otherwise(pl.col("category"))
            .alias("category")
        )
        .with_columns(
            pl.col("category").replace(JBB_CATEGORY_MAPPING_DICT)
        )
        .filter(pl.col("prompt") != "REDACTED")
        .select("prompt", "category")
    )
    return dataset

def split_validation_data(
        data: pl.DataFrame, samples_per_category: int, seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if samples_per_category < 2:
        raise ValueError("validation.samples_per_category must be at least 2")
    generator = random.Random(seed)
    val_idx = []
    for group in data.partition_by("category", maintain_order=True):
        if len(group) < samples_per_category:
            continue # skip small categories
        indices = generator.sample(range(len(group)), min(samples_per_category, len(group) - 1))
        val_idx.append(group[indices])
    if len(val_idx) < 2:
        raise ValueError(f"Validation requires at least two categories with more than {samples_per_category} samples")
    val_data = pl.concat(val_idx)
    train_data = data.join(val_data, on="prompt", how="anti")
    return train_data, val_data


class TokenizedDataset(Dataset[dict[str, int]]):

    def __init__(self,
                 data: pl.DataFrame,
                 tokenizer: AutoTokenizer,
                 max_length: int = 512,
                 prompt_col_name: str = "prompt",
                 category_col_name: str = "category",
                 ) -> None:
        self._category_to_index: dict[str, int] = {
            category: i for i, category in enumerate(data[category_col_name].unique(maintain_order=True))
        }
        self.tokenizer_output = tokenizer(
            data[prompt_col_name].to_list(), padding=False,
            max_length=max_length, return_token_type_ids=False,
        )
        self.category_ids = [self._category_to_index[cat] for cat in data[category_col_name]]

    def __len__(self) -> int:
        return len(self.tokenizer_output["input_ids"])

    def __getitem__(self, index: int) -> dict[str, int]:
        tokenized = {key: values[index] for key, values in self.tokenizer_output.items()}
        category_id = {"category_id": self.category_ids[index]}
        return tokenized | category_id


class StratifiedBatchSampler(Sampler[list[int]]):

    def __init__(
            self,
            strata: Iterable[Hashable],
            cat_number: int,
            samples_per_cat: int,
            seed: int = 42,
    ) -> None:
        if cat_number <= 0 or samples_per_cat <= 0:
            raise ValueError("cat_number and samples_per_cat must be positive")

        groups = {}
        for index, stratum in enumerate(strata):
            groups.setdefault(stratum, []).append(index)

        self.groups = list(groups.values())
        if cat_number > len(self.groups):
            raise ValueError("cat_number cannot exceed the number of categories")

        self.cat_number = cat_number
        self.samples_per_cat = samples_per_cat
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return sum(1 for _ in self._category_batches())

    def _category_batches(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        remaining = torch.tensor([len(group) for group in self.groups], dtype=torch.float64)
        active_count = len(self.groups)
        while active_count:
            categories = torch.multinomial(
                remaining, min(self.cat_number, active_count),
                replacement=False, generator=generator,
            ).tolist()
            for category in categories:
                remaining[category] = max(0, remaining[category].item() - self.samples_per_cat)
                if remaining[category].item() == 0:
                    active_count -= 1
            yield categories

    def __iter__(self) -> Iterator[list[int]]:
        self.epoch += 1
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        shuffled_groups = [
            [group[i] for i in torch.randperm(len(group), generator=generator).tolist()]
            for group in self.groups
        ]
        offsets = [0] * len(self.groups)
        for categories in self._category_batches():
            batch = []
            for category in categories:
                group = shuffled_groups[category]
                start = offsets[category]
                stop = min(start + self.samples_per_cat, len(group))
                batch.extend(group[start:stop])
                offsets[category] = stop
            order = torch.randperm(len(batch), generator=generator).tolist()
            yield [batch[i] for i in order]
