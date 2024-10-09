from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset, Features, load_dataset
from torch import Generator
from torch.nn.functional import pad
from transformers import PreTrainedTokenizerBase


random_generator = Generator()
random_generator = random_generator.manual_seed(2147483647)


def load_dataset_from_files_or_hf(
        type_or_huggingface_path: str = "json",
        filepaths: Optional[Union[str, List[str], Dict[str, Union[str, List[str]]]]] = None, 
        field: Optional[str] = None, split: Optional[str] = None,
        features: Optional[Dict[str, Any]] = None, **kwargs
    ):
    if features:
        features = Features.from_dict(features)
    dataset = load_dataset(
        path=type_or_huggingface_path,
        data_files=filepaths,
        field=field,
        split=split,
        features=features,
        **kwargs
    )
    return dataset


def tokenize_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizerBase, field: Optional[str] = None, 
                     return_tensors: str = "pt", tokenizer_kwargs: Dict[str, Any] = {}, 
                     dataset_kwargs: Dict[str, Any]= {}):
    dataset = dataset.with_transform(
        lambda example: tokenizer(text=example[field] if field else example, **tokenizer_kwargs, return_tensors=return_tensors), 
        **dataset_kwargs
    )
    return dataset


def chunk_example(item, pad_token_id: int, field_name: str = None, chunk_size: Union[Optional[int], Tuple[int, int]] = 4) -> torch.Tensor:
    if field_name:
        example = item[field_name]
    else:
        example = item
    assert len(example.shape) <= 2, f"Assumed input should be of maximum 2 dimensions, got input of shape {example.shape}"
    if len(example.shape) == 2:
        assert example.shape[0] == 1, "Assumed working on single example, not a batch"
    example = example.reshape(1, -1)

    if isinstance(chunk_size, tuple):
        ## Variable chunking
        assert len(chunk_size) == 2
        chunk_size = torch.randint(low=chunk_size[0], high=chunk_size[1], size=(1,), generator=random_generator, dtype=torch.int32).item()
    right_padding = chunk_size - (example.shape[-1] % chunk_size)
    example = pad(input=example, pad=(0, right_padding), mode="constant", value=pad_token_id)
    return example.reshape(1, -1, chunk_size)


def tokenize_and_chunk_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizerBase, 
                               tokenizer_field: Optional[str] = None, chunk_field: Optional[str] = "input_ids",
                               chunk_size: Union[Optional[int], Tuple[int, int]] = 4, 
                               return_tensors: str = "pt", tokenizer_kwargs: Dict[str, Any] = {}, 
                               dataset_kwargs: Dict[str, Any]= {}):
    dataset = dataset.with_transform(
        lambda item: {
            "chunked_input_ids": chunk_example(
                item=tokenizer(item[tokenizer_field] if tokenizer_field else item, return_tensors=return_tensors, **tokenizer_kwargs),
                field_name=chunk_field,
                pad_token_id=tokenizer.pad_token_id,
                chunk_size=chunk_size,
            )},
        **dataset_kwargs
    )
    return dataset
