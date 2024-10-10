from pathlib import Path

import pytest
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ouroboros.utils.data import (
    load_dataset_from_files_or_hf,
    tokenize_dataset,
    apply_prompt_template_to_dataset
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

def test_apply_prompt_template():
    sample_data = load_dataset_from_files_or_hf(type_or_huggingface_path="ybisk/piqa", trust_remote_code=True)
    prompt_dataset = apply_prompt_template_to_dataset(
        prompt_template="Your goal is as follows: {goal}. Pick the option corresponding the right solution.\n(1){sol1}\n(2){sol2}"
    )


# def test_data_loader():
#     sample_data = load_dataset_from_files_or_hf(type_or_huggingface_path="ybisk/piqa", trust_remote_code=True, split="train")
#     tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")

#     fields=["goal", "sol1", "sol2"]

#     tokenized_dataset = tokenize_dataset(
#         sample_data, tokenizer, fields=fields,
#         tokenizer_kwargs={
#             "return_attention_mask": False
#         }
#     )

#     data_collator = CustomDataCollator(
#         tokenizer=tokenizer,
#     )
#     dataloader = DataLoader(
#         dataset=tokenized_dataset,
#         batch_size=16,
#         collate_fn=data_collator,
#     )


#     for i, batch in enumerate(dataloader):
#         inputs = batch["input_ids"]
#         labels = batch["labels"]