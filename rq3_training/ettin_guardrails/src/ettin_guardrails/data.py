import random
from collections.abc import Hashable, Iterable, Iterator

import polars as pl
import torch
from torch.utils.data import Dataset, Sampler
from transformers import AutoTokenizer


def load_data(
        dataset_path: str,
        hf_token: str,
        prompt_col: str = "prompt",
        stratum_col: str = "subcategory",
        label_col: str = "prompt_harm_label",
) -> pl.DataFrame:
    return (
        pl.read_parquet(dataset_path, storage_options={"token": hf_token})
        .group_by(prompt_col, maintain_order=True)
        .agg(
            pl.col(label_col).unique(),
            pl.col(stratum_col).unique()
        )
        .filter((pl.col(stratum_col).list.len() == 1) & (pl.col(label_col).list.len() == 1))
        .filter(pl.col(prompt_col).str.len_bytes() > 0)
        .explode([stratum_col, label_col], empty_as_null=True, keep_nulls=False)
        .select(
            pl.col(prompt_col).alias("prompt"),
            pl.col(stratum_col).alias("category"),
            pl.col(label_col).alias("label")
        ).with_row_index(name="id")
    )

def split_validation_data(
        data: pl.DataFrame,
        fraction: float,
        seed: int,
        index_col: str = "id",
        stratum_col: str = "category",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if fraction <= 0 or fraction >= 1.0:
        raise ValueError("Fraction mast be between 0 and 1")
    val_size = int(data.height * fraction)
    samples_per_cat = (
        data.group_by(stratum_col)
        .agg(((pl.len() / data.height) * val_size).floor().cast(dtype=int).alias("val"))
        .filter(pl.col("val") > 1)
    )
    samples_per_cat = dict(zip(samples_per_cat[stratum_col], samples_per_cat["val"]))
    generator = random.Random(seed)
    val_idx = []
    for category in data.partition_by(stratum_col, maintain_order=True):
        name = category[stratum_col][0]
        if name not in samples_per_cat:
            continue # skip categories with one sample
        indices = generator.sample(range(category.height), samples_per_cat[name])
        val_idx.append(category[indices].select(index_col))
    if len(val_idx) < 2:
        raise ValueError(f"Validation requires at least two categories with more than 2 samples")
    val_data = data.join(pl.concat(val_idx), on=index_col)
    train_data = data.join(val_data, on=index_col, how="anti")
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
