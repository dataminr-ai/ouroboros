import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ouroboros.models import MambaDecoderForCausalLM


def load_dataset(file_path):
    dataset = []
    with open(file_path, "r") as file:
        for line in file:
            dataset.append(json.loads(line))
    return dataset


def update_instance_inputs(instance, prompt):
    instance["inputs"] = prompt + instance["inputs"]
    instance["label_str"] = instance["label_str"]
    return instance


def tokenize_example(example, tokenizer):
    tokenized_inputs = tokenizer(
        example["inputs"],
        return_attention_mask=False,
    )
    tokenized_label = tokenizer(
        example["label_str"],
        return_attention_mask=False,
    )
    input_ids = tokenized_inputs["input_ids"]
    labels = tokenized_label["input_ids"]
    return {"input_ids": input_ids, "labels": labels}


def collate_fn(x):
    # NOTE(rlogan): This is slow but correct
    # TODO: Make the max size configurable instead of 128
    max_seq_len = max(len(x_["input_ids"]) for x_ in x)
    batch_size = len(x)

    input_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.int64)
    for i, x_ in enumerate(x):
        input_ids[i, : len(x_["input_ids"])] = torch.tensor(x_["input_ids"])[
            :max_seq_len
        ]
    labels = torch.full((batch_size, 1), fill_value=-100, dtype=torch.int64)
    for i, x_ in enumerate(x):
        labels[i, 0] = torch.tensor(x_["labels"])
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default=None,
        help="Path to dataset file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Path to dataset file",
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default=None,
        help="Path to dataset file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to dataset file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Path to dataset file",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=200,
        help="Path to dataset file",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Load dataset
    dataset = load_dataset(args.eval_file)
    if args.prompt_dir:
        prompt_file = os.path.join(args.prompt_dir, "prompt.txt")
        with open(prompt_file, "r") as file:
            prompt = file.read()
            print(prompt)
        dataset = [update_instance_inputs(example, prompt) for example in dataset]

    print(dataset[0])
    dataset = [tokenize_example(example, tokenizer) for example in dataset]

    unique_labels = []
    for example in dataset:
        labels = example["labels"]
        unique_labels.extend(labels)
    unique_labels = list(set(unique_labels))
    print(unique_labels)

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Model
    model = MambaDecoderForCausalLM.from_pretrained(args.model_name, use_mambapy=True)
    model.eval()
    model.cuda()
    device = "cuda"

    correct = 0
    total = 0

    for idx, batch in enumerate(data_loader):
        print(idx)
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch["input_ids"])
        next_token_logits = outputs.logits[:, -1, :]
        mask = torch.full_like(
            next_token_logits, float("-inf")
        )  # Fill with very negative values to mask out unwanted logits
        mask[:, unique_labels] = next_token_logits[
            :, unique_labels
        ]  # Keep only logits 17 and 18

        # Get the argmax over the restricted logits
        preds = mask.argmax(dim=-1)

        correct += (preds == batch["labels"][:, 0]).sum().item()
        total += len(batch["labels"])

    accuracy = correct / total
    print(f"Accuracy: {accuracy}")

    if args.output_dir:
        output_dir = args.output_dir
        accuracy_file = os.path.join(output_dir, "accuracy.txt")
        with open(accuracy_file, "w") as file:
            file.write(f"{accuracy}")


if __name__ == "__main__":
    main()
