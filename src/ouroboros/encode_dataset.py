import argparse
from transformers import AutoConfig, MambaForCausalLM, AutoTokenizer
import torch
import json
import pickle
import gc
import os


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
        """
        padding_length = block_size - len(chunks["input_ids"][-1])
        chunks["input_ids"][-1] = torch.cat(
            [chunks["input_ids"][-1], torch.zeros(padding_length, dtype=torch.long)]
        )  # pad_token_id is 0
        chunks["attention_mask"][-1] = torch.cat(
            [
                chunks["attention_mask"][-1],
                torch.zeros(padding_length, dtype=torch.long),
            ]
        )
        """
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


def move_cache(cache_params, device):
    for key in cache_params.ssm_states:
        cache_params.ssm_states[key] = cache_params.ssm_states[key].to(device)
    for key in cache_params.conv_states:
        cache_params.conv_states[key] = cache_params.conv_states[key].to(device)
    return cache_params


def get_cache_params(batch, model):
    outputs = model(batch, output_hidden_states=True, use_cache=True, return_dict=True)
    hidden_states = outputs.cache_params
    #hidden_states.conv_states.zero_()
    return hidden_states


def main(
    input_file,
    model_id,
    config_path,
    chunk_size,
    batch_size,
    output_file,
    checkpoints=None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(config_path)
    model = MambaForCausalLM.from_pretrained(model_id, config=config)
    model.cuda()

    raw_dataset = read_dataset(input_file)
    tokenized_dataset = tokenize_dataset(raw_dataset, tokenizer)
    chunked_dataset = chunk_dataset(tokenized_dataset, chunk_size)
    batched_chunks = batch_chunks(
        chunked_dataset, batch_size
    )  # batched_chunks[batch_number]['input_ids'][instance_number]
    print("Number of batches: ", len(batched_chunks))
    encoded_dataset = batched_chunks.copy()

    for idx, batch in enumerate(batched_chunks):
        print(idx)
        with torch.no_grad():
            input_ids = batch["input_ids"].to("cuda")
            hidden_states = get_cache_params(input_ids, model)
            encoded_dataset[idx]["cache_params"] = hidden_states
        del hidden_states, input_ids
        gc.collect()
        torch.cuda.empty_cache()

        completed_steps = idx + 1
        if checkpoints and (idx + 1) % checkpoints == 0:
            # Save encoded_dataset as pickle file
            output_path = os.path.join(output_file, f"subset_{idx + 1}.pkl")
            with open(output_path, "wb") as f:
                pickle.dump(
                    encoded_dataset[
                        completed_steps - checkpoints : completed_steps - 1
                    ],
                    f,
                )
            encoded_dataset[completed_steps - checkpoints : completed_steps - 1] = [
                None
            ] * (checkpoints - 1)
            print(f"Saved and cleared checkpoint {completed_steps}")

    if not checkpoints:
        # Save encoded_dataset as pickle file
        with open(output_file, "wb") as f:
            pickle.dump(encoded_dataset, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process dataset and get hidden states."
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Input dataset filename (jsonl)"
    )
    parser.add_argument("--model_id", type=str, required=True, help="Model ID")
    parser.add_argument("--config_path", type=str, required=True, help="Config.json")
    parser.add_argument(
        "--output_file", type=str, required=True, help="Output dataset filename (pkl)"
    )
    parser.add_argument(
        "--chunk_size", type=int, required=True, help="Size of each chunk"
    )

    parser.add_argument(
        "--checkpoints", type=int, help="Divide dataset into checkpoints if needed"
    )

    parser.add_argument(
        "--batch_size", type=int, required=True, help="Size of each batch"
    )
    args = parser.parse_args()

    main(
        args.input_file,
        args.model_id,
        args.config_path,
        args.chunk_size,
        args.batch_size,
        args.output_file,
        args.checkpoints,
    )
