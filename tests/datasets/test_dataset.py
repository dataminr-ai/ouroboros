from pathlib import Path

import pytest
from transformers import AutoTokenizer
import torch

import ouroboros.encode_dataset as ed
from ouroboros.utils.data import load_dataset_from_files_or_hf, tokenize_dataset


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
        }, 
        dataset_kwargs={"batched": True}
    )
    assert "input_ids" in tokenized_dataset.column_names
    assert "attention_masks" not in tokenized_dataset.column_names

    for i, input_id in enumerate(tokenized_dataset["input_ids"]):
        tokenized = tokenizer(train_data["text"][i], return_tensors="pt", return_attention_mask=False)["input_ids"]
        ## since tokenization was done as batched 
        ## we check if tokenization is correct upto where padding begins
        assert torch.allclose(
            tokenized[0], 
            torch.tensor(input_id[:tokenized[0].shape[0]], dtype=torch.int64, requires_grad=False)
        )


def test_chunking_dataset():
    sample_train_data_file = Path(__file__).parent / "samples" / "sample_train.jsonl"
    data = ed.read_dataset(sample_train_data_file)
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    tokenized_data = ed.tokenize_dataset(dataset=data, tokenizer=tokenizer)
    chunked_data = ed.chunk_dataset(
        tokenized_dataset=tokenized_data, 
        block_size=4
    )
    assert chunked_data is not None