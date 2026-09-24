import random

import polars as pl
from huggingface_hub import get_token
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, BatchEncoding


def load_data(
        hf_dataset_path: str,
        prompt_col: str = "prompt",
        stratum_col: str = "subcategory",
        label_col: str = "prompt_harm_label",
        harmful_label: str = "harmful"
) -> pl.DataFrame:
    return (
        pl.read_parquet(hf_dataset_path, storage_options={"token": get_token()})
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
            (pl.col(label_col) == harmful_label).alias("label")
        ).with_row_index(name="id")
    )


def split_validation_data(
        data: pl.DataFrame,
        fraction: float,
        index_col: str = "id",
        stratum_col: str = "category",
        seed: int = 42,
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
            continue  # skip categories with one sample
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
                 tokenizer: PreTrainedTokenizerBase,
                 max_length: int = 7999,
                 prompt_col: str = "prompt",
                 stratum_col: str = "category",
                 label_col: str = "label",
                 ) -> None:
        self._stratum_to_index: dict[str, int] = {
            category: i for i, category in enumerate(data[stratum_col].unique(maintain_order=True))
        }
        self._data = data
        self._tokenizer: PreTrainedTokenizerBase = tokenizer
        self._stratum_col = stratum_col
        self._prompt_col = prompt_col
        self._label_col = label_col
        self._max_length = max_length

    @property
    def strata_labels(self):
        return [self._stratum_to_index[value] for value in self._data[self._stratum_col]]

    def __len__(self) -> int:
        return self._data.height

    def __getitem__(self, index: int) -> BatchEncoding:
        prompt = self._data[self._prompt_col][index]
        category = self._data[self._stratum_col][index]
        tokenizer_output = self._tokenizer(
            prompt,
            padding=False,
            truncation=True,
            max_length=self._max_length,
            return_token_type_ids=False
        )
        tokenizer_output["category_ids"] = self._stratum_to_index[category]
        tokenizer_output["labels"] = self._data[self._label_col][index]
        return tokenizer_output

    def __getitems__(self, indices: list[int]):
        prompts = self._data[self._prompt_col][indices]
        categories = self._data[self._stratum_col][indices].map_elements(lambda x: self._stratum_to_index[x])
        tokenizer_output = self._tokenizer(
            prompts.to_list(),
            padding=False,
            truncation=True,
            max_length=self._max_length,
            return_token_type_ids=False
        )
        tokenizer_output["category_ids"] = categories.to_list()
        tokenizer_output["labels"] = self._data[self._label_col][indices].to_list()
        return tokenizer_output
