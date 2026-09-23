import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl
from polars.testing import assert_frame_equal

from ettin_guardrails.data import load_data, split_validation_data


class LoadDataTests(unittest.TestCase):
    def load_rows(self, rows, **kwargs):
        columns = [
            kwargs.get("prompt_col", "prompt"),
            kwargs.get("stratum_col", "subcategory"),
            kwargs.get("label_col", "prompt_harm_label"),
        ]
        source = pl.DataFrame(
            rows, schema={column: pl.String for column in columns}, orient="row"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.parquet"
            source.write_parquet(path)
            return load_data(str(path), hf_token="test-token", **kwargs)

    def test_deduplicates_and_preserves_order_labels_and_schema(self):
        result = self.load_rows([
            ("second", "b", "harmful"),
            ("first", "a", "harmless"),
            ("second", "b", "harmful"),
            ("third", "a", "unrecognized"),
        ])
        expected = pl.DataFrame({
            "prompt": ["second", "first", "third"],
            "category": ["b", "a", "a"],
            "label": ["harmful", "harmless", "unrecognized"],
        }).with_row_index("id")
        assert_frame_equal(result, expected)

    def test_removes_conflicting_annotations_and_empty_prompts(self):
        result = self.load_rows([
            ("label conflict", "a", "harmful"),
            ("label conflict", "a", "harmless"),
            ("category conflict", "a", "harmful"),
            ("category conflict", "b", "harmful"),
            ("", "a", "harmful"),
            (None, "a", "harmful"),
            ("valid", "b", "harmless"),
        ])
        self.assertEqual(result.rows(), [(0, "valid", "b", "harmless")])

    def test_preserves_whitespace_only_prompts(self):
        result = self.load_rows([(" \t", "a", "harmless")])
        self.assertEqual(result["prompt"].to_list(), [" \t"])

    def test_supports_custom_column_names(self):
        result = self.load_rows(
            [("hello", "greeting", "harmless")],
            prompt_col="text", stratum_col="topic", label_col="annotation",
        )
        self.assertEqual(result.columns, ["id", "prompt", "category", "label"])
        self.assertEqual(result.rows(), [(0, "hello", "greeting", "harmless")])

    def test_all_filtered_rows_return_empty_frame_with_output_schema(self):
        result = self.load_rows([("", "a", "harmful")])
        expected = pl.DataFrame(schema={
            "prompt": pl.String, "category": pl.String, "label": pl.String,
        }).with_row_index("id")
        assert_frame_equal(result, expected)

    def test_passes_path_and_hugging_face_token_to_reader(self):
        source = pl.DataFrame({
            "prompt": ["hello"], "subcategory": ["a"],
            "prompt_harm_label": ["harmless"],
        })
        with patch("ettin_guardrails.data.pl.read_parquet", return_value=source) as reader:
            load_data("hf://datasets/example/data.parquet", "test-token")
        reader.assert_called_once_with(
            "hf://datasets/example/data.parquet", storage_options={"token": "test-token"}
        )


class SplitValidationDataTests(unittest.TestCase):
    @staticmethod
    def make_data(sizes):
        categories = [name for name, count in sizes.items() for _ in range(count)]
        return pl.DataFrame({
            # Nonconsecutive IDs ensure sampling positions aren't mistaken for IDs.
            "id": [100 + 3 * i for i in range(len(categories))],
            "category": categories,
            "prompt": [f"prompt {i}" for i in range(len(categories))],
            "label": ["harmful"] * len(categories),
        })

    def test_floors_allocations_and_keeps_small_strata_in_training(self):
        data = self.make_data({"a": 13, "b": 9, "small": 4, "singleton": 1})
        train, validation = split_validation_data(data, fraction=0.3, seed=42)
        # Target size is 8; allocations floor to 3, 2, 1, 0, then 1 is skipped.
        counts = dict(validation.group_by("category").len().iter_rows())
        self.assertEqual(counts, {"a": 3, "b": 2})
        self.assertEqual(train.height, 22)
        self.assertTrue(set(train["id"]).isdisjoint(validation["id"]))
        self.assertEqual(validation["id"].n_unique(), validation.height)
        assert_frame_equal(pl.concat([train, validation]).sort("id"), data.sort("id"))

    def test_same_seed_reproduces_split_without_changing_input_or_global_rng(self):
        data = self.make_data({"a": 20, "b": 20})
        original = data.clone()
        rng_state = random.getstate()
        train, validation = split_validation_data(data, 0.3, seed=42)
        repeated_train, repeated_validation = split_validation_data(data, 0.3, seed=42)
        assert_frame_equal(train.sort("id"), repeated_train.sort("id"))
        assert_frame_equal(validation.sort("id"), repeated_validation.sort("id"))
        assert_frame_equal(data, original)
        self.assertEqual(random.getstate(), rng_state)

    def test_supports_custom_id_and_stratum_columns(self):
        data = self.make_data({"a": 10, "b": 10})
        train, validation = split_validation_data(data, 0.4, seed=7)
        renamed = data.rename({"id": "row_id", "category": "topic"})
        custom_train, custom_validation = split_validation_data(
            renamed, 0.4, seed=7, index_col="row_id", stratum_col="topic"
        )
        for expected, actual in [(train, custom_train), (validation, custom_validation)]:
            assert_frame_equal(
                expected.sort("id"),
                actual.rename({"row_id": "id", "topic": "category"}).sort("id"),
            )

    def test_rejects_fractions_outside_open_unit_interval(self):
        data = self.make_data({"a": 10, "b": 10})
        for fraction in [-1.0, 0.0, 1.0, 1.5]:
            with self.subTest(fraction=fraction):
                with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                    split_validation_data(data, fraction, seed=42)

    def test_requires_two_strata_with_at_least_two_validation_samples(self):
        for sizes in [{"a": 10}, {"a": 10, "b": 4}, {"a": 4, "b": 4}]:
            with self.subTest(sizes=sizes):
                with self.assertRaisesRegex(ValueError, "at least two categories"):
                    split_validation_data(self.make_data(sizes), 0.3, seed=42)


if __name__ == "__main__":
    unittest.main()
