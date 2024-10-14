from pathlib import Path

import pytest
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from ouroboros.utils.data import (
    load_dataset_from_files_or_hf,
    tokenize_dataset,
    tokenize_dataset_with_prompt_template,
)


def test_load_multiline_json_file():
    sample_train_data_file = Path(__file__).parent / "samples" / "sample_train.jsonl"
    train_data = load_dataset_from_files_or_hf(
        filepaths=str(sample_train_data_file),
        split="train",
        streaming=False,
    )
    assert train_data.num_rows == 2
    assert "text" in train_data.features


@pytest.mark.xfail(raises=ValueError)
def test_data_load_failure():
    sample_train_data_file = Path(__file__).parent / "samples" / "sample_train.jsonl"
    _ = load_dataset_from_files_or_hf(
        filepaths=str(sample_train_data_file),
        split="test"
    )

def test_load_piqa():
    sample_data = load_dataset_from_files_or_hf(type_or_huggingface_path="ybisk/piqa", trust_remote_code=True)
    assert sample_data["train"].dataset_size > 0
    assert sample_data["test"].column_names == ["goal", "sol1", "sol2", "label"]

def test_load_lambada():
    sample_data = load_dataset_from_files_or_hf(type_or_huggingface_path="cimec/lambada", trust_remote_code=True)
    assert sample_data["train"].dataset_size > 0
    assert sample_data["test"].column_names == ["text", "domain"]

def test_tokenized_dataload_for_lm():
    sample_train_data_file = Path(__file__).parent / "samples" / "sample_train.jsonl"
    sample_test_data_file = Path(__file__).parent / "samples" / "sample_eval.jsonl"
    batch_size = 16

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
        tokenizer=tokenizer, dataset=train, keep_only_relevant_columns={"input_ids"}
    )

    dataloader = DataLoader(
        dataset=tokenized_dataset,
        batch_size=batch_size,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    for batch in dataloader:
        assert "input_ids" in batch
        assert batch["input_ids"].shape[0] == 2


def test_encode_with_template_for_evaluation():
    batch_size = 4
    feature_fields = ["goal", "sol1", "sol2"]
    prompt_template = "Your goal is as follows: {goal}. Pick the option corresponding the right solution.\n(1){sol1}\n(2){sol2}"
    sample_data = load_dataset_from_files_or_hf(type_or_huggingface_path="ybisk/piqa", trust_remote_code=True, split="test")
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    tokenized_dataset = tokenize_dataset_with_prompt_template(
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        dataset=sample_data,
        feature_fields=feature_fields,
        apply_chat_template=False
    )    
    dataloader = DataLoader(
        dataset=tokenized_dataset,
        batch_size=batch_size,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    smol_data = list(sample_data.take(batch_size * 4))

    for i, batch in enumerate(dataloader):
        assert "input_ids" in batch
        assert batch["input_ids"].shape[0] == batch_size
        decoded = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)

        for idx, item in enumerate(decoded):
            raw = smol_data[i*batch_size + idx]
            real = prompt_template.format(**raw)
            assert real == item
        
        if i == 3:
            break
