from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from transformers.utils import PaddingStrategy

from ouroboros.utils.data import (
    DataCollatorForContrastiveLMTraining,
    load_dataset_from_files_or_hf,
    tokenize_dataset,
)


def test_load_multiline_json_file():
    sample_train_data_file = Path(__file__).parent / "samples" / "sample_train.jsonl"
    train_data = load_dataset_from_files_or_hf(
        filepaths=str(sample_train_data_file),
        split="train",
        streaming=False,
    )
    assert train_data.num_rows > 0
    assert "inputs" in train_data.features


@pytest.mark.xfail(raises=ValueError)
def test_data_load_failure():
    sample_train_data_file = Path(__file__).parent / "samples" / "sample_train.jsonl"
    _ = load_dataset_from_files_or_hf(
        filepaths=str(sample_train_data_file),
        split="test"
    )


def test_tokenized_dataload_for_mamba_seq2seq():
    sample_train_data_file = Path(__file__).parent / "samples" / "sample_train.jsonl"
    sample_test_data_file = Path(__file__).parent / "samples" / "sample_eval.jsonl"
    batch_size = 4

    train = load_dataset_from_files_or_hf(
        filepaths={
            "train": str(sample_train_data_file),
            "test": str(sample_test_data_file)
        },
        split="train",
        streaming=True
    )

    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")

    tokenized_dataset = tokenize_dataset(
        tokenizer=tokenizer, dataset=train, training=True
    )

    sample_train = list(train.take(batch_size*4))

    dataloader = DataLoader(
        dataset=tokenized_dataset,
        batch_size=batch_size,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer)
    )

    for i, batch in enumerate(dataloader):
        if i == 4: 
            break
        assert "input_ids" in batch
        assert batch["input_ids"].shape[0] == 4
        decoded_tokens = tokenizer.batch_decode(torch.where(batch["labels"] == -100, tokenizer.pad_token_id, batch["labels"]), skip_special_tokens=True)
        for idx, dt in enumerate(decoded_tokens):
            label = sample_train[i*batch_size + idx]["label_str"]
            assert dt == label



def test_tokenized_dataload_for_contrastive_training():
    sample_train_data_file = Path(__file__).parent / "samples" / "sample_train.jsonl"
    sample_test_data_file = Path(__file__).parent / "samples" / "sample_eval.jsonl"
    batch_size = 4

    train = load_dataset_from_files_or_hf(
        filepaths={
            "train": str(sample_train_data_file),
            "test": str(sample_test_data_file)
        },
        split="train",
        streaming=True
    )

    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    tokenizer.pad
    tokenizer.model_input_names = ["positive_input_ids", "negative_input_ids"]
    tokenized_dataset = tokenize_dataset(
        tokenizer=tokenizer, dataset=train, contrastive=True
    )

    sample_train = list(train.take(batch_size*4))

    dataloader = DataLoader(
        dataset=tokenized_dataset,
        batch_size=batch_size,
        collate_fn=DataCollatorForContrastiveLMTraining(tokenizer=tokenizer)
    )

    for i, batch in enumerate(dataloader):
        if i == 4: 
            break
        assert "positive_input_ids" in batch
        assert batch["positive_input_ids"].shape[0] == 4
        decoded_positive_ids = tokenizer.batch_decode(batch["positive_input_ids"], skip_special_tokens=True)
        decoded_negative_ids = tokenizer.batch_decode(batch["negative_input_ids"], skip_special_tokens=True)
        for idx, (dt, dn) in enumerate(zip(decoded_positive_ids, decoded_negative_ids)):
            positive = sample_train[i*batch_size + idx]["positive"]
            negative = sample_train[i*batch_size + idx]["negative"]

            assert dt == positive
            assert dn == negative


def test_tokenized_dataload_with_max_len_for_contrastive_training():
    sample_train_data_file = Path(__file__).parent / "samples" / "sample_train.jsonl"
    sample_test_data_file = Path(__file__).parent / "samples" / "sample_eval.jsonl"
    batch_size = 4
    max_seq_len = 127

    train = load_dataset_from_files_or_hf(
        filepaths={
            "train": str(sample_train_data_file),
            "test": str(sample_test_data_file)
        },
        split="train",
        streaming=True
    )

    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    tokenizer.pad
    tokenizer.model_input_names = ["positive_input_ids", "negative_input_ids"]
    tokenized_dataset = tokenize_dataset(
        tokenizer=tokenizer, dataset=train, contrastive=True
    )

    dataloader = DataLoader(
        dataset=tokenized_dataset,
        batch_size=batch_size,
        collate_fn=DataCollatorForContrastiveLMTraining(tokenizer=tokenizer, max_length=max_seq_len, padding=PaddingStrategy.MAX_LENGTH if max_seq_len else True),
    )

    for i, batch in enumerate(dataloader):
        if i == 4: 
            break
        assert "positive_input_ids" in batch
        assert batch["positive_input_ids"].shape[0] == 4
        assert batch["positive_input_ids"].shape[-1] == max_seq_len
        assert "negative_input_ids" in batch
        assert batch["negative_input_ids"].shape[-1] == max_seq_len
        assert batch["negative_input_ids"].shape[0] == 4
