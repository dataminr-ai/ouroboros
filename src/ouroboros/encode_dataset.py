import json
import math
import random

import torch


def read_dataset(filename):
    texts = []
    with open(filename, "r") as f:
        for line in f:
            data = json.loads(line)
            texts.append(data["text"])
    return texts


def tokenize_dataset(dataset, tokenizer):
    tokenized = []
    for item in dataset:
        tokens = tokenizer(item, return_tensors="pt")
        tokenized.append(tokens)
    return tokenized


def chunk_dataset(tokenized_dataset, block_size):
    block_size = block_size  # Make room for bos_token
    concatenated_input_ids = torch.cat(
        [tokens["input_ids"][0] for tokens in tokenized_dataset], dim=0
    )

    total_length = len(concatenated_input_ids)
    chunks = {
        "input_ids": [
            concatenated_input_ids[i : i + block_size]
            for i in range(0, total_length, block_size)
        ],
    }
    # Add padding for last chunk if less than block_size
    block_size = block_size + 1
    if len(chunks["input_ids"][-1]) < block_size:
        chunks["input_ids"].pop()
    return chunks


def chunk_dataset_varied(tokenized_dataset, min_block_size=4, max_block_size=64):
    concatenated_input_ids = torch.cat(
        [tokens["input_ids"][0] for tokens in tokenized_dataset], dim=0
    )

    total_length = len(concatenated_input_ids)
    chunks = {"input_ids": []}

    i = 0
    while i < total_length:
        # Random block size between min_block_size and max_block_size
        #block_size = random.randint(min_block_size, max_block_size)
        min_exp = math.ceil(math.log2(min_block_size))
        max_exp = math.floor(math.log2(max_block_size))
        block_size = 2 ** random.randint(min_exp, max_exp)
        end_index = min(i + block_size, total_length)
        chunks["input_ids"].append(concatenated_input_ids[i:end_index])
        i = end_index

    # Remove any empty last chunk (in case total_length == 0)
    if len(chunks["input_ids"][-1]) == 0:
        chunks["input_ids"].pop()

    return chunks


def batch_chunks(chunked_dataset, batch_size):
    total_chunks = len(chunked_dataset["input_ids"])
    batched_chunks = []
    for i in range(0, total_chunks, batch_size):
        batched_chunks.append(
            {
                "input_ids": torch.stack(
                    chunked_dataset["input_ids"][i : i + batch_size]
                ),
            }
        )
    return batched_chunks


def batch_chunks_varied(chunked_dataset, batch_size):
    pad_token=0
    total_chunks = len(chunked_dataset["input_ids"])
    batched_chunks = []

    for i in range(0, total_chunks, batch_size):
        batch = chunked_dataset["input_ids"][i : i + batch_size]

        # Find the maximum length in this batch
        max_length = max(len(chunk) for chunk in batch)

        # Pad each chunk in the batch to max_length
        padded_batch = [
            torch.cat([chunk, torch.full((max_length - len(chunk),), pad_token)])
            if len(chunk) < max_length else chunk
            for chunk in batch
        ]

        # Stack the padded chunks into a single tensor for the batch
        batched_chunks.append(
            {
                "input_ids": torch.stack(padded_batch),
            }
        )

    return batched_chunks


def move_cache(cache_params, device):
    for key in cache_params.ssm_states:
        cache_params.ssm_states[key] = cache_params.ssm_states[key].to(device)
    for key in cache_params.conv_states:
        cache_params.conv_states[key] = cache_params.conv_states[key].to(device)
    return cache_params


def get_cache_params(batch, model):
    outputs = model(batch, output_hidden_states=True, use_cache=True, return_dict=True)
    hidden_states = outputs.cache_params
    return hidden_states
