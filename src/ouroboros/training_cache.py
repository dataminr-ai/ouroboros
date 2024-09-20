#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    MODEL_MAPPING,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
)
from transformers.cache_utils import MambaCache
from transformers.models.mamba import MambaConfig
from transformers.models.mamba.modeling_mamba import is_fast_path_available
from transformers.utils.versions import require_version

import ouroboros.encode_dataset as ed
from ouroboros.decode_cache import reconstruct
from ouroboros.models import (
    MambaDecoderConfig,
    MambaDecoderForCausalLM,
    TrainableMambaCache,
)

require_version(
    "datasets>=2.14.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)
AutoModelForCausalLM.register(MambaDecoderConfig, MambaDecoderForCausalLM)

logger = logging.getLogger(__name__)
logging.getLogger("py4j").setLevel(logging.DEBUG)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="Path to dataset file",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=200,
        help="Maximum sequence length for training data",
        required=False,
    )
    parser.add_argument(
        "--decoder",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
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

    parser.add_argument(
        "--reconstructor",
        type=str,
        help="Check point for reconstructor",
        required=False,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )

    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--no_keep_linebreaks",
        action="store_true",
        help="Do not keep line breaks when using TXT files.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_token", type=str, help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training",
    )
    parser.add_argument(
        "--starting_prompt",
        type=str,
        default=None,
        help="Prompt to initiate the cache",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )

    args = parser.parse_args()

    return args


def save_checkpoint(model, optimizer, scheduler, epoch, step, checkpoint_path):
    os.makedirs(checkpoint_path, exist_ok=True)
    # print("Made directory")    #model.save_pretrained(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, "training_state.bin")
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved at epoch {epoch}, step {step}")


def load_dataset(file_path):
    dataset = []
    with open(file_path, "r") as file:
        for line in file:
            dataset.append(json.loads(line))
    return dataset


def tokenize_example(example, tokenizer):
    tokenized_inputs = tokenizer(
        example["inputs"],
        return_attention_mask=False,
    )
    tokenized_label = tokenizer(
        example["label_str"],
        return_attention_mask=False,
    )
    input_ids = tokenized_inputs["input_ids"] + tokenized_label["input_ids"]
    labels = [-100] * len(tokenized_inputs["input_ids"]) + tokenized_label["input_ids"]
    return {"input_ids": input_ids, "labels": labels}


def collate_fn(x, max_len=200):
    # NOTE(rlogan): This is slow but correct
    # TODO: Make the max size configurable instead of 128
    max_seq_len = min(max(len(x_["input_ids"]) for x_ in x), max_len)
    batch_size = len(x)

    input_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.int64)
    for i, x_ in enumerate(x):
        input_ids[i, : len(x_["input_ids"])] = torch.tensor(x_["input_ids"])[
            :max_seq_len
        ]
    labels = torch.full((batch_size, max_seq_len), fill_value=-100, dtype=torch.int64)
    for i, x_ in enumerate(x):
        labels[i, : len(x_["labels"])] = torch.tensor(x_["labels"])[:max_seq_len]
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    if is_fast_path_available:
        logger.info("Fast path is available.")
    else:
        logger.info(
            "Fast path is not available. Enabling will greatly speed up encoding."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.decoder,
        trust_remote_code=args.trust_remote_code,
    )

    model = MambaDecoderForCausalLM.from_pretrained(args.decoder, use_mambapy=True)
    model.to(args.device)
    model.train()

    # Load Dataset
    dataset = load_dataset(args.train_file)
    dataset = [tokenize_example(example, tokenizer) for example in dataset]
    # dataset = dataset.map(lambda example: tokenize_example(example, tokenizer))

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, args.max_seq_len),
    )

    # Initialize cache
    if args.starting_prompt is not None:
        prompt = ["Pick the best option that answers the question.\n"]
        token_prompt = tokenizer(prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            encoded_prompt = ed.get_cache_params(token_prompt["input_ids"], model)
        learned_conv_state = encoded_prompt.conv_states
        learned_ssm_state = encoded_prompt.ssm_states
    else:
        learned_conv_state = None
        learned_ssm_state = None

    encoder_cache_params = TrainableMambaCache(
        config=model.config,
        batch_size=args.batch_size,
        learned_conv_state=learned_conv_state,
        learned_ssm_state=learned_ssm_state,
        device=args.device,
        dtype=model.dtype,
    )

    params_to_optimize = [{"params": encoder_cache_params.parameters()}]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(dataset) / args.gradient_accumulation_steps / args.batch_size
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # TODO(rlogan): Add back checkpointing
    completed_steps, start_step = 0, 0

    # For reconstruction
    if args.reg:
        config = MambaConfig(args.decoder)

        decoder = MambaDecoderForCausalLM.from_pretrained(
            args.reconstructor, torch_dtype=model.dtype
        )
        decoder.eval()
        decoder.to(args.device)

    with tqdm(total=args.max_train_steps, desc="Training Progress") as pbar:
        pbar.update(completed_steps)
        for epoch in range(0, args.num_train_epochs):
            for step, batch in enumerate(train_loader):
                if step > start_step:
                    batch = {k: v.to(args.device) for k, v in batch.items()}

                    if batch["input_ids"].shape[0] == args.batch_size:
                        logger.info("Step: " + str(completed_steps))
                        outputs = model(
                            **batch,
                            encoder_cache_params=encoder_cache_params,
                        )

                        if not args.reg:
                            loss = outputs.loss
                        else:
                            # get learned hidden state...

                            learned_cache_params = MambaCache(
                                config=config, max_batch_size=1, dtype=model.dtype
                            )
                            learned_cache_params.conv_states = (
                                encoder_cache_params.learned_conv_state.detach().clone()
                            )
                            learned_cache_params.ssm_states = (
                                encoder_cache_params.learned_ssm_state.detach().clone()
                            )

                            # reconstructed state encoder(decoder(learned_cache_params))
                            decoded_cache = reconstruct(
                                decoder, tokenizer, learned_cache_params
                            ).to(args.device)
                            with torch.no_grad():
                                recon_cache_params = ed.get_cache_params(
                                    decoded_cache, model
                                )

                            # define distance function
                            ssm_dist = torch.norm(
                                learned_cache_params.ssm_states
                                - recon_cache_params.ssm_states
                            )
                            conv_dist = torch.norm(
                                learned_cache_params.conv_states
                                - recon_cache_params.conv_states
                            )

                            # Loss
                            loss = outputs.loss + args.reg_strength * (
                                ssm_dist + conv_dist
                            )

                        logger.info("Loss: " + str(loss.item()))
                        logger.info(
                            "Memory: " + str(torch.cuda.memory_allocated()) + "\n"
                        )
                        loss.backward()

                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                        completed_steps += 1
                        pbar.update(1)

                        if completed_steps % checkpointing_steps == 0:
                            output_dir = f"step_{completed_steps}"
                            if args.output_dir is not None:
                                output_dir = os.path.join(args.output_dir, output_dir)
                            logging.info(
                                "Saving Checkpoint at Step"
                                + str(completed_steps)
                                + " in directory "
                                + str(output_dir)
                            )
                            # model.save_pretrained(output_dir)
                            save_checkpoint(
                                encoder_cache_params,
                                optimizer,
                                lr_scheduler,
                                epoch,
                                step,
                                output_dir,
                            )
    output_dir = os.path.join(args.output_dir, f"step_{completed_steps}")
    # model.save_pretrained(output_dir)
    save_checkpoint(
        encoder_cache_params, optimizer, lr_scheduler, epoch, step, output_dir
    )
    logging.info(
        "Saving final checkpoint for epoch "
        + str(epoch)
        + "in directory "
        + str(output_dir)
    )


if __name__ == "__main__":
    main()
