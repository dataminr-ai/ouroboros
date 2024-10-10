from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset, Features, IterableDataset, load_dataset
from torch import Generator
from torch.nn.functional import pad
from transformers import PreTrainedTokenizerBase


random_generator = Generator()
random_generator = random_generator.manual_seed(2147483647)


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

def apply_prompt_template_to_example(
        tokenizer: PreTrainedTokenizerBase,
        prompt_template: str,
        example: Dict[str, str],
        feature_fields: List[str],
        label_field: Optional[str] = None, 
        apply_chat_template: bool = True,
        training: bool = False, 
        add_generation_prompt: bool = False
    ):
    format_items = {
        f: example[f] for f in feature_fields
    }
    label = None

    if label_field:
        label = example[label_field]

    prompt = prompt_template.format(**format_items)
    return_dict = {}

    if not apply_chat_template:
        return_dict["texts"] = prompt
        if label:
            return_dict["labels"] = label
    else:
        chat = [
            {"role": "user", "content": prompt}
        ]
        if training and label:
            chat.append(
                {"role": "assistant", "content": label}
            )
        
        return_dict["texts"] = tokenizer.apply_chat_template(
            conversation=chat,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )
        if label:
            return_dict["labels"] = label

    return return_dict

def apply_prompt_template_to_dataset(
        tokenizer: PreTrainedTokenizerBase,
        prompt_template: str, 
        dataset: IterableDataset,
        feature_fields: List[str],
        label_field: str,
        apply_chat_template: bool = True,
        training: bool = False, 
        add_generation_prompt: bool = False
    ):
    return (
        dataset.map(
            lambda example: apply_prompt_template_to_example(
                tokenizer=tokenizer,
                prompt_template=prompt_template,
                example=example,
                feature_fields=feature_fields,
                label_field=label_field,
                apply_chat_template=apply_chat_template,
                training=training,
                add_generation_prompt=add_generation_prompt
            )
            .with_format(format=format)
        )
    )

def tokenize_example(tokenizer: PreTrainedTokenizerBase, example: Dict[str, Any], text_field: str = "texts", return_remaining_fields: bool = False):
    outputs = {}
    output = tokenizer(example[text_field], return_attention_mask=False, return_tensors="pt")
    if isinstance(output, Mapping):
        outputs["input_ids"] = output["input_ids"]
    else:
        outputs["input_ids"] = output

    if return_remaining_fields:
        for key in example.keys():
            if key == text_field:
                continue
            if key not in outputs:
                outputs[key] = example[key]
    return outputs


def tokenize_dataset(tokenizer: PreTrainedTokenizerBase, dataset: Dataset, text_field: str = "texts", return_remaining_fields: bool = False):
    dataset = dataset.map(
        lambda example: tokenize_example(
            tokenizer=tokenizer,
            example=example,
            text_field=text_field,
            return_remaining_fields=return_remaining_fields
        ),
    )
    return dataset


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

def chunk_example(item: Dict[str, Any], pad_token_id: int, field_name: Union[str, List[str]], chunk_field_name: str = "input_ids", chunk_size: Union[Optional[int], Tuple[int, int]] = 4) -> torch.Tensor:
    outputs = {}
    if isinstance(field_name, str):
        example = item[field_name]
        if isinstance(example, Mapping):
            example = example[chunk_field_name]
        outputs[field_name] = _chunk_tensor(example, pad_token_id=pad_token_id, chunk_size=chunk_size)
    else:
        for field in field_name:
            outputs[field] = chunk_example(item, pad_token_id=pad_token_id, field_name=field, chunk_size=chunk_size).get(field)
    return outputs


def tokenize_and_chunk_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizerBase, 
                               tokenizer_fields: List[str], chunk_field_name: str = "input_ids",
                               chunk_size: Union[Optional[int], Tuple[int, int]] = 4, 
                               return_tensors: str = "pt", tokenizer_kwargs: Dict[str, Any] = {}, 
                               dataset_kwargs: Dict[str, Any]= {}):
    dataset = dataset.with_transform(
        lambda item: chunk_example(
                item=tokenize_example(
                    tokenizer=tokenizer,
                    example=item,
                    fields=tokenizer_fields,
                    return_tensors=return_tensors,
                    **tokenizer_kwargs
                ),
                field_name=tokenizer_fields,
                chunk_field_name=chunk_field_name,
                pad_token_id=tokenizer.pad_token_id,
                chunk_size=chunk_size,
            ),
        **dataset_kwargs
    )
    return dataset
