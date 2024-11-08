import argparse
import functools
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ouroboros.cache_utils import (
    collate_fn,
    contrastive_collate_fn,
    contrastive_tokenize_example,
    load_dataset,
    tokenize_example,
    validate_classification,
    validate_contrastive,
)
from ouroboros.models import MambaDecoderForCausalLM, TrainableMambaCache


def update_instance_inputs(instance, prompt):
    instance["inputs"] = prompt + instance["inputs"]
    instance["label_str"] = instance["label_str"]
    return instance

def update_instance_contrastive(instance, prompt):
    instance["positive"] = prompt + instance["positive"]
    instance["negative"] = prompt + instance["negative"]
    return instance

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
        "--output_path",
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
        default=None,
        help="Path to dataset file",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Path to dataset file",
    )
    parser.add_argument("--contrastive", action="store_true")
    parser.add_argument(
        "--validation_limit",
        type=int,
        default=-1,
        help="Limits the number of validation batches (for development)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',

    )
    parser.add_argument(
        "--dpo_weight",
        type=int,
        default=1
    )
    parser.add_argument(
        "--add_eos",
        action="store_true",
        help="Whether to add EOS token to end of labels.",
    )
    parser.add_argument(
        "--reg",
        type=bool,
        help="Whether to use regularization",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--reg_strength",
        type=float,
        help="Strength of regularization",
        default=0,
        required=False,
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
        if args.contrastive:
            dataset = [update_instance_contrastive(example, prompt) for example in dataset]
            print(dataset[0])
        else:
            dataset = [update_instance_inputs(example, prompt) for example in dataset]
            print(dataset[0])

    if args.contrastive:
        tokenize_fn = contrastive_tokenize_example
        collate_fn_ = contrastive_collate_fn
    else:
        tokenize_fn = functools.partial(tokenize_example, add_eos=args.add_eos)
        collate_fn_ = collate_fn

    dataset = [tokenize_fn(example, tokenizer) for example in dataset]
    valid_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: collate_fn_(batch, args.max_seq_len),
        )

    # Model
    model = MambaDecoderForCausalLM.from_pretrained(args.model_name, use_mambapy=True)
    model.to(args.device, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.eval()

    if args.cache_dir:
        encoder_cache_params = TrainableMambaCache(config=model.config, dtype=torch.bfloat16)
        state_dict = torch.load(os.path.join(args.cache_dir, 'training_state.bin'))
        encoder_cache_params.load_state_dict(state_dict["model_state_dict"])
        encoder_cache_params.cuda()
    else:
        encoder_cache_params = None

    if args.contrastive:
        _, valid_acc = validate_contrastive(
            valid_loader, model, args, encoder_cache_params
        )
    else:
        if args.reg:
            #valid_loss, valid_acc = validate_classification(
             #   valid_loader, model, args, encoder_cache_params, config, tokenizer, decoder
            #)
            print ("Not implemented")
        else:
            _, valid_acc = validate_classification(
                valid_loader, model, args, encoder_cache_params
            )

    print("Validation Acc: " , str(valid_acc))


if __name__ == "__main__":
    main()
