from pathlib import Path

from ouroboros import encode_dataset as ed


def test_read_dataset():
    sample_train_data_file = Path(__file__).parent / "samples" / "sample_train.jsonl"
    train_samples = ed.read_dataset(filename=sample_train_data_file)

    assert len(train_samples) == 2
    for sample in train_samples:
        assert isinstance(sample, str)


def 