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
import logging
import math
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    MODEL_MAPPING,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
)
from transformers.models.mamba.modeling_mamba import is_fast_path_available

import ouroboros.encode_dataset as ed
from ouroboros.models import MambaDecoderConfig, MambaDecoderForCausalLM


logger = logging.getLogger(__name__)
logging.getLogger("py4j").setLevel(logging.ERROR)


AutoModelForCausalLM.register(MambaDecoderConfig, MambaDecoderForCausalLM)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
torch.autograd.set_detect_anomaly(True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A json file containing the training data",
    )
    # TODO(rlogan): Use
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv, txt or a json file containing the validation data.",
    )
    # TODO(rlogan): Use
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--decoder",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    # TODO(rlogan): Use or lose
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
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
    # TODO(rlogan): Use or lose
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    # TODO(rlogan): Use with datasets when added back.
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    # TODO(rlogan): Use with datasets when added back.
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
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
        "--chunk_size",
        type=int,
        default = None,
        help="Sequence Length for fixed sequence length training",
    )
    parser.add_argument(
        "--mixed_chunk",
        type=bool,
        default = False,
        help="Whether to use mixed chunk sizes",
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
    # TODO(rlogan): Add tracking
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
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    model.save_pretrained(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, "training_state.bin")
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved at epoch {epoch}, step {step}")


def main():
    args = parse_args()

    if is_fast_path_available:
        logger.info('Fast path is available.')
    else:
        logger.info('Fast path is not available. Enabling will greatly speed up encoding.')

    tokenizer = AutoTokenizer.from_pretrained(
        args.decoder,
        use_fast=not args.use_slow_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )

    if not args.resume_from_checkpoint:
        model = MambaDecoderForCausalLM.from_pretrained(
            args.decoder,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
            use_mambapy=True,
        )
    elif args.resume_from_checkpoint:
        checkpoint_path = os.path.join(args.output_dir, 'step_'+ args.resume_from_checkpoint)
        model = MambaDecoderForCausalLM.from_pretrained(
            checkpoint_path,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
            use_mambapy=True,
        )
        state_path = os.path.join(checkpoint_path, 'training_state.bin')
        checkpoint = torch.load(state_path)
    logger.info(model.config.to_dict())
    logger.info(model)
    model.train()
    model.to(args.device)

    encoder = AutoModelForCausalLM.from_pretrained(
        args.encoder,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        trust_remote_code=args.trust_remote_code,
    )
    encoder.eval()
    encoder.to(args.device)

    # Load Dataset
    raw_dataset = ed.read_dataset(args.train_file)
    tokenized_dataset = ed.tokenize_dataset(raw_dataset, tokenizer)

    if not args.mixed_chunk:
        chunked_dataset = ed.chunk_dataset(tokenized_dataset, args.chunk_size)
        batched_chunks = ed.batch_chunks(
            chunked_dataset, args.batch_size
        )
    else:
        chunked_dataset = ed.chunk_dataset_varied(tokenized_dataset)
        batched_chunks = ed.batch_chunks_varied(
        chunked_dataset, args.batch_size
    )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(batched_chunks) / args.gradient_accumulation_steps
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

    # Load from checkpoint if available
    if not args.resume_from_checkpoint:
        completed_steps, start_step = 0, 0
    elif args.resume_from_checkpoint:
        completed_steps, start_step = int(args.resume_from_checkpoint), int(args.resume_from_checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    with tqdm(total=args.max_train_steps, desc="Training Progress") as pbar:
        pbar.update(completed_steps)
        for epoch in range(0, args.num_train_epochs):
            for step, batch in enumerate(batched_chunks):
                if step > start_step:
                    batch = {k: v.to(args.device) for k, v in batch.items()}
                    logger.info("Step: " + str(completed_steps))
                    logger.info("Encode")
                    with torch.no_grad():
                        cache_params = ed.get_cache_params(batch['input_ids'], encoder)
                    logger.info(batch['input_ids'].device)
                    logger.info(cache_params)

                    logger.info("Forward")
                    input_ids = F.pad(batch['input_ids'], (1, 1), value=tokenizer.eos_token_id)
                    logger.info(batch['input_ids'].device)
                    outputs = model(
                        input_ids=input_ids,
                        encoder_cache_params=cache_params,
                        return_dict=True,
                        labels=input_ids,
                    )
                    loss = outputs.loss

                    logger.info("Loss: " + str(loss.item()))
                    logger.info("Memory: " + str(torch.cuda.memory_allocated()) + "\n")

                    logger.info("Backprop")
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
                        save_checkpoint(
                            model, optimizer, lr_scheduler, epoch, step, output_dir
                        )
                    if completed_steps >= args.max_train_steps:
                        break

    output_dir = os.path.join(args.output_dir, f"step_{completed_steps}")
    save_checkpoint(model, optimizer, lr_scheduler, epoch, step, output_dir)
    logging.info(
        "Saving final checkpoint for epoch "
        + str(epoch)
        + "in directory "
        + str(output_dir)
    )


if __name__ == "__main__":
    main()
