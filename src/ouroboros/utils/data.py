from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from datasets import Dataset, Features, IterableDataset, load_dataset
from torch import Generator
from torch.nn.functional import pad
from transformers import PreTrainedTokenizerBase


random_generator = Generator()
random_generator = random_generator.manual_seed(2147483647)

__all__ = ["load_dataset_from_files_or_hf", "tokenize_dataset", "chunk_tokenized_dataset"]


def load_dataset_from_files_or_hf(
        type_or_huggingface_path: str = "json",
        filepaths: Optional[Union[str, List[str], Dict[str, Union[str, List[str]]]]] = None, 
        split: Optional[str] = None,
        streaming: bool = True,
        features: Optional[Dict[str, Any]] = None, 
        cache_dir: Optional[str] = str(Path.cwd() / "cache"),
        **kwargs
    ):
    if features:
        features = Features.from_dict(features)
    dataset = load_dataset(
        path=type_or_huggingface_path,
        data_files=filepaths,
        split=split,
        streaming=streaming,
        features=features,
        cache_dir=cache_dir,
        **kwargs
    )

    return dataset

def apply_prompt_to_dataset(dataset: Dataset, prompt: str):
    dataset.map(
        lambda example: {
            "inputs": prompt + example["inputs"]
        }
    )

def tokenize_example(example: Dict[str, Any], tokenizer: PreTrainedTokenizerBase, label_pad_token_id: int = -100, training: bool = False):
    tokenized_inputs = tokenizer(
        example["inputs"],
        return_attention_mask=False
    )
    tokenized_label = None
    if "label_str" in example:
        tokenized_label = tokenizer(
            example["label_str"],
            return_attention_mask=False,
        )
    
    if training and tokenized_label:
        input_ids = tokenized_inputs["input_ids"] + tokenized_label["input_ids"]
        labels = [label_pad_token_id] * len(tokenized_inputs["input_ids"]) + tokenized_label["input_ids"]
    else:
        input_ids = tokenized_inputs["input_ids"]
        labels = tokenized_label["input_ids"] if tokenized_label else None

    return {"input_ids": input_ids, "labels": labels}

def tokenize_dataset(
        tokenizer: PreTrainedTokenizerBase,
        dataset: IterableDataset,
        training: bool = False,
        contrastive: bool = False,
    ):
    tokenized_dataset = dataset.map(
        lambda example: tokenize_example(tokenizer=tokenizer, example=example, training=training)
    )
    if contrastive:
        tokenized_dataset = tokenized_dataset.select_columns(["input_ids", "positive", "negative"])
    if not contrastive:
        tokenized_dataset = tokenized_dataset.select_columns(["input_ids", "labels"])
    return tokenized_dataset


def chunk_tokenized_dataset(dataset: Dataset, pad_token_id: int, input_field: str = "input_ids", chunk_size: Union[Optional[int], Tuple[int, int]] = 4):
    chunked_dataset = dataset.map(
        lambda example: _chunk_example(example, pad_token_id=pad_token_id, field_name=input_field, chunk_size=chunk_size)
    )
    return chunked_dataset


def _chunk_tensor(tensor: Union[torch.Tensor, List[int]], pad_token_id: int, chunk_size: Union[Optional[int], Tuple[int, int]] = 4):
    if isinstance(tensor, list):
        tensor = torch.tensor(tensor, requires_grad=False)
    assert isinstance(tensor, torch.Tensor), f"Invalid type for tensor object: {type(tensor)}"
    assert len(tensor.shape) <= 2, f"Assumed input should be of maximum 2 dimensions, got input of shape {tensor.shape}"
    if len(tensor.shape) == 2:
        assert tensor.shape[0] == 1, "Assumed working on single tensor, not a batch"
    tensor = tensor.reshape(1, -1)

    if isinstance(chunk_size, tuple):
        ## Variable chunking
        assert len(chunk_size) == 2
        chunk_size = torch.randint(low=chunk_size[0], high=chunk_size[1], size=(1,), generator=random_generator, dtype=torch.int32).item()
    right_padding = chunk_size - (tensor.shape[-1] % chunk_size)
    tensor = pad(input=tensor, pad=(0, right_padding), mode="constant", value=pad_token_id)
    return tensor.reshape(-1, chunk_size)

def _chunk_example(item: Dict[str, Any], pad_token_id: int, field_name: str, chunk_size: Union[Optional[int], Tuple[int, int]] = 4) -> torch.Tensor:
    outputs = {}
    example = item[field_name]
    if isinstance(example, Mapping):
        example = example[field_name]
    outputs[field_name] = _chunk_tensor(example, pad_token_id=pad_token_id, chunk_size=chunk_size)
    return outputs
