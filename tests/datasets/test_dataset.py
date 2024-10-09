from pathlib import Path

import pytest

from ouroboros.utils.data import load_dataset_from_files_or_hf


def test_load_multiline_json_file():
    sample_train_data_file = Path(__file__).parent / "samples" / "sample_train.jsonl"
    train_data = load_dataset_from_files_or_hf(
        filepaths=str(sample_train_data_file),
        split="train"
    )
    assert train_data.num_rows == 2
    assert "text" in train_data.column_names


@pytest.mark.xfail(raises=ValueError)
def test_data_load_failure():
    sample_train_data_file = Path(__file__).parent / "samples" / "sample_train.jsonl"
    _ = load_dataset_from_files_or_hf(
        filepaths=str(sample_train_data_file),
        split="test"
    )


def test_load_piqa():
    sample_data = load_dataset_from_files_or_hf(type_or_huggingface_path="ybisk/piqa", trust_remote_code=True)
    assert sample_data["train"].num_rows > 0