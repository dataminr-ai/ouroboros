from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from ouroboros.utils.data import (
    load_dataset_from_files_or_hf,
    tokenize_and_chunk_dataset,
    tokenize_dataset,
)


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


def test_tokenization():
    sample_train_data_file = Path(__file__).parent / "samples" / "sample_train.jsonl"
    train_data = load_dataset_from_files_or_hf(
        filepaths=str(sample_train_data_file),
        split="train"
    )
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    tokenized_dataset = tokenize_dataset(
        dataset=train_data, 
        tokenizer=tokenizer,
        field="text", 
        tokenizer_kwargs={
            "return_attention_mask": False,
            "padding": True,
            "truncation": True,
            "max_length": 4096
        }
    )

    for i, item in enumerate(tokenized_dataset):
        assert "input_ids" in item
        tokenized = tokenizer(train_data["text"][i], return_tensors="pt", return_attention_mask=False)["input_ids"]
        ## since tokenization was done as batched 
        ## we check if tokenization is correct upto where padding begins
        assert torch.allclose(
            tokenized[0], 
            item["input_ids"]
        )


def test_chunked_dataset():
    sample_train_data_file = Path(__file__).parent / "samples" / "sample_train.jsonl"
    train_data = load_dataset_from_files_or_hf(
        filepaths=str(sample_train_data_file),
        split="train"
    )
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    chunk_size = 4

    tokenized_dataset = tokenize_dataset(
        dataset=train_data, 
        tokenizer=tokenizer,
        field="text", 
        tokenizer_kwargs={
            "return_attention_mask": False,
            "padding": True,
            "truncation": True,
            "max_length": 4096
        }
    )

    chunked_dataset = tokenize_and_chunk_dataset(
        dataset=train_data, 
        tokenizer=tokenizer,
        tokenizer_field="text", 
        tokenizer_kwargs={
            "return_attention_mask": False,
            "padding": True,
            "truncation": True,
            "max_length": 4096
        },
        chunk_size=chunk_size,
    )

    for chunked_item, tokenized_item in zip(chunked_dataset, tokenized_dataset):
        assert "chunked_input_ids" in chunked_item
        item_elements = chunked_item["chunked_input_ids"].numel()
        assert chunked_item["chunked_input_ids"].shape[-1] == chunk_size
        assert chunk_size - (tokenized_item["input_ids"].shape[-1] % chunk_size) + tokenized_item["input_ids"].shape[-1] == item_elements


def test_variable_chunked_dataset():
    sample_train_data_file = Path(__file__).parent / "samples" / "sample_train.jsonl"
    train_data = load_dataset_from_files_or_hf(
        filepaths=str(sample_train_data_file),
        split="train"
    )
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    chunk_size = (4, 64)

    tokenized_dataset = tokenize_dataset(
        dataset=train_data, 
        tokenizer=tokenizer,
        field="text", 
        tokenizer_kwargs={
            "return_attention_mask": False,
            "padding": True,
            "truncation": True,
            "max_length": 4096
        }
    )

    chunked_dataset = tokenize_and_chunk_dataset(
        dataset=train_data, 
        tokenizer=tokenizer,
        tokenizer_field="text", 
        tokenizer_kwargs={
            "return_attention_mask": False,
            "padding": True,
            "truncation": True,
            "max_length": 4096
        },
        chunk_size=chunk_size,
    )

    for chunked_item, tokenized_item in zip(chunked_dataset, tokenized_dataset):
        assert "chunked_input_ids" in chunked_item
        item_elements = chunked_item["chunked_input_ids"].numel()
        item_chunk_size = chunked_item["chunked_input_ids"].shape[-1]
        assert item_chunk_size - (tokenized_item["input_ids"].shape[-1] % item_chunk_size) + tokenized_item["input_ids"].shape[-1] == item_elements