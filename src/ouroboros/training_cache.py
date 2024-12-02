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
import functools
import logging
import math
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    MODEL_MAPPING,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
)
from transformers.models.mamba import MambaConfig
from transformers.models.mamba.modeling_mamba import is_fast_path_available

import ouroboros.encode_dataset as ed
from ouroboros.cache_utils import (
    classification_loss_outputs,
    collate_fn,
    contrastive_accuracy_loss,
    contrastive_collate_fn,
    contrastive_tokenize_example,
    load_dataset,
    tokenize_example,
    validate_classification,
    validate_contrastive,
)
from ouroboros.models import (
    MambaDecoderConfig,
    MambaDecoderForCausalLM,
    TrainableMambaCache,
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
        "--validation_file",
        type=str,
        default=None,
        help="Path to dataset file",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=10,
        help="Number of steps between each validation run",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default=None,
        help="Path to dataset file",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
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
    # TODO(rlogan): Use or lose
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    # TODO(rlogan): Use or lose
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    # TODO(rlogan): Use or lose
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
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--add_eos",
        action="store_true",
        help="Whether to add EOS token to end of labels.",
    )
    parser.add_argument(
        "--validation_limit",
        type=int,
        default=-1,
        help="Limits the number of validation batches (for development)",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=1, help="Logging frequency"
    )
    parser.add_argument(
        "--dpo_weight", type=int, default=1, help="Logging frequency"
    )
    parser.add_argument("--contrastive", action="store_true")

    args = parser.parse_args()

    return args


def save_checkpoint(model, optimizer, scheduler, epoch, step, checkpoint_path):
    os.makedirs(checkpoint_path, exist_ok=True)
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


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    if is_fast_path_available:
        logger.info("Fast path is available.")
    else:
        logger.info(
            "Fast path is not available. Enabling will greatly speed up encoding."
        )
    summary_writer = SummaryWriter(log_dir=args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        args.decoder,
        trust_remote_code=args.trust_remote_code,
    )

    model = MambaDecoderForCausalLM.from_pretrained(args.decoder, use_mambapy=True, ignore_mismatched_sizes=True)
    #model.to(args.device, dtype=torch.bfloat16)
    model.to(args.device)
    model.gradient_checkpointing_enable()
    model.train()

    # Load Dataset
    if args.contrastive:
        tokenize_fn = contrastive_tokenize_example
        collate_fn_ = contrastive_collate_fn
    else:
        tokenize_fn = functools.partial(tokenize_example, add_eos=args.add_eos)
        collate_fn_ = collate_fn
    dataset = load_dataset(args.train_file)
    dataset = [tokenize_fn(example, tokenizer) for example in dataset]

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_(batch, args.max_seq_len),
    )

    if args.validation_file:
        validation_dataset = load_dataset(args.validation_file)
        validation_dataset = [
            tokenize_fn(example, tokenizer) for example in validation_dataset
        ]
        valid_loader = DataLoader(
            validation_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: collate_fn_(batch, args.max_seq_len),
        )
    if args.eval_file:
        evaluation_dataset = load_dataset(args.eval_file)
        evaluation_dataset = [
            tokenize_fn(example, tokenizer) for example in evaluation_dataset
        ]
        eval_loader = DataLoader(
            evaluation_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: collate_fn_(batch, args.max_seq_len),
        )

    # Initialize cache
    if not args.resume_from_checkpoint:
        if args.starting_prompt is not None:
                prompt = [args.starting_prompt]
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
    elif args.resume_from_checkpoint:
        checkpoint_path = os.path.join(args.output_dir, 'step_'+ args.resume_from_checkpoint)
        encoder_cache_params = TrainableMambaCache(config=model.config, dtype=model.dtype)
        state_dict = torch.load(os.path.join(checkpoint_path, 'training_state.bin'))
        encoder_cache_params.load_state_dict(state_dict["model_state_dict"])
        encoder_cache_params.to(args.device)
    
      
    params_to_optimize = [{"params": encoder_cache_params.parameters()}]
    optimizer = torch.optim.AdamW(
        params_to_optimize, lr=args.learning_rate, weight_decay=args.weight_decay
    )

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

    if not args.resume_from_checkpoint:
        completed_steps, start_step = 0, 0
    elif args.resume_from_checkpoint:
        completed_steps, start_step = int(args.resume_from_checkpoint), int(args.resume_from_checkpoint)
        optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        lr_scheduler.load_state_dict(state_dict["scheduler_state_dict"])

    # For reconstruction
    if args.reg:
        config = MambaConfig(args.decoder)

        decoder = MambaDecoderForCausalLM.from_pretrained(
            args.reconstructor, torch_dtype=model.dtype
        )
        decoder.eval()
        decoder.to(args.device)

    # Base Model Validation and Evaluation
    '''
    if completed_steps == 0:
        if args.contrastive:
            valid_loss, valid_acc = validate_contrastive(
            valid_loader, model, args, encoder_cache_params
        )
        else:
            if args.reg:
                valid_loss, valid_acc = validate_classification(
                    valid_loader, model, args, encoder_cache_params, config, tokenizer, decoder
                )
            else:
                valid_loss, valid_acc = validate_classification(
                    valid_loader, model, args
                )
        logger.info("Validation Loss: " + str(valid_loss))
        logger.info("Validation Acc: " + str(valid_acc))
        summary_writer.add_scalar(
            "Loss/valid", valid_loss, completed_steps
        )
        summary_writer.add_scalar(
            "Acc/valid", valid_acc, completed_steps
        )
        if args.eval_file:
            if args.contrastive:
                eval_loss, eval_acc = validate_contrastive(
                eval_loader, model, args, encoder_cache_params
                )
            else:
                if args.reg:
                    eval_loss, eval_acc = validate_classification(
                        eval_loader, model, args, encoder_cache_params, config, tokenizer, decoder
                    )
                else:
                    eval_loss, eval_acc = validate_classification(
                        eval_loader, model, args
                    )
            logger.info("Eval Loss: " + str(eval_loss))
            logger.info("Eval Acc: " + str(eval_acc))
            summary_writer.add_scalar(
            "Loss/test", eval_loss, completed_steps
            )
            summary_writer.add_scalar(
            "Acc/test", eval_acc, completed_steps
            )
'''
    model.train()

    with tqdm(total=args.max_train_steps, desc="Training Progress") as pbar:
        pbar.update(completed_steps)
        for epoch in range(0, args.num_train_epochs):
            for step, batch in enumerate(train_loader):
                if step > start_step:
                    batch = {k: v.to(args.device) for k, v in batch.items()}
                    batch_size = next(iter(batch.values())).size(0)
                    encoder_cache_params.resize(batch_size)
                    if args.contrastive:
                        if args.reg:
                            _ , _ , loss = contrastive_accuracy_loss(batch, model, args, encoder_cache_params, config, tokenizer, decoder)
                        else:
                            _ , _ , loss = contrastive_accuracy_loss(batch, model, args, encoder_cache_params)
                    else:
                        if args.reg:
                            loss, _ = classification_loss_outputs(batch, model, encoder_cache_params, args, config, tokenizer, decoder)
                        else:
                            loss, _ = classification_loss_outputs(batch, model, encoder_cache_params, args)

                    loss.backward()

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    completed_steps += 1
                    pbar.update(1)
                    if completed_steps % args.logging_steps == 0:
                        summary_writer.add_scalar(
                            "Loss/train", loss.item(), completed_steps
                        )
                        summary_writer.add_scalar(
                            "Memory",
                            torch.cuda.memory_allocated(args.device),
                            completed_steps,
                        )

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
                            encoder_cache_params,
                            optimizer,
                            lr_scheduler,
                            epoch,
                            step,
                            output_dir,
                        )

                    if completed_steps % args.validation_steps == 0:
                        if args.contrastive:
                            if args.reg:
                                valid_loss, valid_acc = validate_contrastive(
                                    valid_loader, model, args, encoder_cache_params,  config, tokenizer, decoder
                                )
                            else:
                                valid_loss, valid_acc = validate_contrastive(
                                valid_loader, model, args, encoder_cache_params
                            )
                        else:
                            if args.reg:
                                valid_loss, valid_acc = validate_classification(
                                    valid_loader, model, args, encoder_cache_params, config, tokenizer, decoder
                                )
                            else:
                                valid_loss, valid_acc = validate_classification(
                                    valid_loader, model, args, encoder_cache_params
                                )
                        summary_writer.add_scalar(
                            "Loss/valid", valid_loss, completed_steps
                        )
                        summary_writer.add_scalar(
                            "Acc/valid", valid_acc, completed_steps
                        )
                        if args.eval_file:
                            if args.contrastive:
                                if args.reg:
                                    eval_loss, eval_acc = validate_contrastive(
                                        eval_loader, model, args, encoder_cache_params, config, tokenizer, decoder
                                    )
                                else:
                                    eval_loss, eval_acc = validate_contrastive(
                                eval_loader, model, args, encoder_cache_params
                                )
                            else:
                                if args.reg:
                                    eval_loss, eval_acc = validate_classification(
                                        eval_loader, model, args, encoder_cache_params, config, tokenizer, decoder
                                    )
                                else:
                                    eval_loss, eval_acc = validate_classification(
                                        eval_loader, model, args, encoder_cache_params
                                    )
                            logger.info("Test Loss: " + str(eval_loss))
                            logger.info("Test Acc: " + str(eval_acc))
                            summary_writer.add_scalar(
                            "Loss/test", eval_loss, completed_steps
                            )
                            summary_writer.add_scalar(
                            "Acc/test", eval_acc, completed_steps
                            )
                        model.train()
    output_dir = os.path.join(args.output_dir, f"step_{completed_steps}")
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
