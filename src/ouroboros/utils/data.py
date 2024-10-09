from typing import Any, Dict, List, Optional, Union

from datasets import Dataset, Features, load_dataset
from transformers import PreTrainedTokenizerBase


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
                     tokenizer_kwargs: Dict[str, Any] = {}, dataset_kwargs: Dict[str, Any]= {}):
    dataset = dataset.map(
        lambda example: tokenizer(text=example[field] if field else example, **tokenizer_kwargs), 
        **dataset_kwargs
    )
    return dataset
