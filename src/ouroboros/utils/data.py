from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from datasets import Dataset, Features, IterableDataset, load_dataset
from torch import Generator
from torch.nn.functional import pad
from transformers import PreTrainedTokenizerBase


random_generator = Generator()
random_generator = random_generator.manual_seed(2147483647)

__all__ = ["load_dataset_from_files_or_hf", "tokenize_dataset_with_prompt_template", "tokenize_dataset", "chunk_tokenized_dataset"]


def load_dataset_from_files_or_hf(
        type_or_huggingface_path: str = "json",
        filepaths: Optional[Union[str, List[str], Dict[str, Union[str, List[str]]]]] = None, 
        split: Optional[str] = None,
        streaming: bool = True,
        features: Optional[Dict[str, Any]] = None, **kwargs
    ):
    if features:
        features = Features.from_dict(features)
    dataset = load_dataset(
        path=type_or_huggingface_path,
        data_files=filepaths,
        split=split,
        streaming=streaming,
        features=features,
        **kwargs
    )

    return dataset

def _tokenize_using_prompt_template(
        tokenizer: PreTrainedTokenizerBase,
        example: Dict[str, str],
        prompt_template: str,
        feature_fields: List[str],
        apply_chat_template: bool = True,
        training: bool = False,
        label_field: str = "label",
        add_generation_prompt: bool = False
    ):
    format_items = {
        f: example[f] for f in feature_fields
    }

    prompt = prompt_template.format(**format_items)

    if not apply_chat_template:
        output = tokenizer(prompt, return_attention_mask=False)
    else:
        chat = [
            {"role": "user", "content": prompt}
        ]
        if training and label_field:
            chat.append(
                {"role": "assistant", "content": str(example[label_field])}
            )
        
        output = tokenizer.apply_chat_template(
            conversation=chat,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            tokenizer_kwargs={
                "return_attention_mask": False
            }
        )
    return output

def tokenize_dataset_with_prompt_template(
        tokenizer: PreTrainedTokenizerBase,
        prompt_template: str, 
        dataset: IterableDataset,
        feature_fields: List[str],
        label_field: str = "label",
        apply_chat_template: bool = True,
        training: bool = False, 
        add_generation_prompt: bool = False,
        keep_only_relevant_columns: Union[bool, Set[str]] = True,
    ):
    tokenized_dataset =  (
        dataset.map(
            lambda example: _tokenize_using_prompt_template(
                tokenizer=tokenizer,
                prompt_template=prompt_template,
                example=example,
                feature_fields=feature_fields,
                label_field=label_field,
                apply_chat_template=apply_chat_template,
                training=training,
                add_generation_prompt=add_generation_prompt
            )
        )
    )
    if keep_only_relevant_columns:
        tokenized_dataset = _select_relevant_columns_from_dataset(
            tokenized_dataset=tokenized_dataset, 
            keep_only_relevant_columns=keep_only_relevant_columns,
            label_field=label_field
        )
    return tokenized_dataset

def _select_relevant_columns_from_dataset(
        tokenized_dataset: Union[IterableDataset, Dataset], 
        keep_only_relevant_columns: Union[Set[str], bool] = True, 
        label_field: str = "label") -> Union[Dataset, IterableDataset]:
    if isinstance(keep_only_relevant_columns, set):
        col_names = list(keep_only_relevant_columns)
    else:
        relevant_columns = {"input_ids", label_field}
        try:
            if tokenized_dataset.features:
                col_names = [col for col in tokenized_dataset.features if col in relevant_columns]
            else:
                tokenized_dataset = tokenized_dataset._resolve_features()
                col_names = [col for col in tokenized_dataset.features if col in relevant_columns]
        except Exception:
            raise RuntimeError(
                "Inferring relevant column names for Dataset, "
                "set relevant_columns manually using keep_only_relevant_columns"
            )

    tokenized_dataset = tokenized_dataset.select_columns(
        column_names=col_names
    )
    return tokenized_dataset

def tokenize_dataset(
        tokenizer: PreTrainedTokenizerBase,
        dataset: IterableDataset, 
        input_field: str = "text",
        label_field: str = "label", 
        keep_only_relevant_columns: Union[bool, Set[str]] = True,
    ):
    tokenized_dataset = dataset.map(
        lambda example: tokenizer(example[input_field], return_attention_mask=False)
    )

    if keep_only_relevant_columns:
        tokenized_dataset = _select_relevant_columns_from_dataset(
            tokenized_dataset=tokenized_dataset, 
            keep_only_relevant_columns=keep_only_relevant_columns,
            label_field=label_field
        )
    return tokenized_dataset

def chunk_tokenized_dataset(dataset: Dataset, pad_token_id: int, input_field: str = "input_ids", chunk_size: Union[Optional[int], Tuple[int, int]] = 4):
    chunked_dataset = dataset.map(
        lambda example: _chunk_example(example, pad_token_id=pad_token_id, field_name=input_field, chunk_size=chunk_size)
    )
    return chunked_dataset


def _chunk_tensor(tensor: torch.Tensor, pad_token_id: int, chunk_size: Union[Optional[int], Tuple[int, int]] = 4):
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
    return tensor.reshape(1, -1, chunk_size)

def _chunk_example(item: Dict[str, Any], pad_token_id: int, field_name: str, chunk_size: Union[Optional[int], Tuple[int, int]] = 4) -> torch.Tensor:
    outputs = {}
    example = item[field_name]
    if isinstance(example, Mapping):
        example = example[field_name]
    outputs[field_name] = _chunk_tensor(example, pad_token_id=pad_token_id, chunk_size=chunk_size)
    return outputs
