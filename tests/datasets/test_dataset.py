from pathlib import Path

import pytest

from ouroboros.utils.data import load_dataset_from_files


def test_load_multiline_json_file():
    sample_train_data_file = Path(__file__).parent / "samples" / "sample_train.jsonl"
    train_data = load_dataset_from_files(
        type="json",
        filepaths=str(sample_train_data_file),
        split="train"
    )
    assert len(train_data["text"]) == 2


@pytest.mark.xfail(raises=ValueError)
def test_data_load_failure():
    sample_train_data_file = Path(__file__).parent / "samples" / "sample_train.jsonl"
    _ = load_dataset_from_files(
        type="json",
        filepaths=str(sample_train_data_file),
        split="test"
    )

