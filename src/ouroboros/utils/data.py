from typing import Any, Dict, List, Optional, Union

from datasets import Features, load_dataset, Dataset
from transformers import PreTrainedTokenizerBase


def load_dataset_from_files(
        filepaths: Union[str, List[str], Dict[str, Union[str, List[str]]]], 
        type: str = "json", field: Optional[str] = None, split: Optional[str] = None,
        features: Optional[Dict[str, Any]] = None,
    ):
    if features:
        features = Features.from_dict(features)
    dataset = load_dataset(
        type,
        data_files=filepaths,
        field=field,
        split=split,
        features=features
    )
    return dataset


def tokenize_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizerBase):
    dataset = dataset.map()
    return dataset
