import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.cache_utils import MambaCache

from ouroboros.models import MambaDecoderForCausalLM, TrainableMambaCache


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
    labels = [-100] * len(tokenized_inputs["input_ids"]) + tokenized_label["input_ids"]
    return {"input_ids": input_ids, "labels": labels}


def collate_fn(x, max_len=200):
    max_seq_len = min(max(len(x_["labels"]) for x_ in x), max_len)
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

def tokenize_contrastive_example(example, tokenizer):
    tokenized_inputs = tokenizer(
        example["inputs"],
        return_attention_mask=False,
    )
    tokenized_sol1 = tokenizer(
        example["sol1"],
        return_attention_mask=False,
    )
    tokenized_sol2 = tokenizer(
        example["sol2"],
        return_attention_mask=False,
    )
    tokenized_label = tokenizer(
        example["label_str"],
        return_attention_mask=False,
    )
    sol1=tokenized_inputs["input_ids"] +tokenized_sol1["input_ids"] 
    sol2=tokenized_inputs["input_ids"] +tokenized_sol2["input_ids"] 
    return {"sol1": sol1, "sol2": sol2, "label": tokenized_label["input_ids"]}


def collate_contrastive_fn(x, max_len=200):
    max_seq_len_1 = min(max(len(x_["sol1"]) for x_ in x), max_len)
    max_seq_len_2 = min(max(len(x_["sol2"]) for x_ in x), max_len)
    batch_size = len(x)

    sol1 = torch.zeros((batch_size, max_seq_len_1), dtype=torch.int64)
    for i, x_ in enumerate(x):
        sol1[i, : len(x_["sol1"])] = torch.tensor(x_["sol1"])[
            :max_seq_len_1
        ]
    sol2 = torch.zeros((batch_size, max_seq_len_2), dtype=torch.int64)
    for i, x_ in enumerate(x):
        sol2[i, : len(x_["sol2"])] = torch.tensor(x_["sol2"])[
            :max_seq_len_2
        ]
    labels = torch.full((batch_size, 1), fill_value=-100, dtype=torch.int64)
    for i, x_ in enumerate(x):
        labels[i, 0] = torch.tensor(x_["label"])
    return {
        "sol1": sol1,
        "sol2": sol2,
        "label": labels,
    }

def calculate_correct(logits, labels):
    # Restrict logits to label tokens
    unique_labels=[17,18]
    vocab_size = logits.size(-1)
    mask = torch.full((vocab_size,), float('-inf'), device=logits.device)  
    mask[unique_labels] = 0 
    restricted_logits = logits + mask

    # Predictions over restricted tokens
    predictions = torch.argmax(restricted_logits, dim=-1)
    print(predictions)
    # Find the last non -100 label for each sequence (last token label)
    flat_labels = labels.view(-1)
    # Remove the -100 values
    flat_labels = flat_labels[flat_labels != -100]
    print(flat_labels)
    correct_predictions = (predictions == flat_labels).sum().item()
    print('Number correct: ', correct_predictions)
    return correct_predictions

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
        default=200,
        help="Path to dataset file",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Path to dataset file",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
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

    if args.metric == 'contrastive':
        dataset = [tokenize_contrastive_example(example, tokenizer) for example in dataset]
        data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_contrastive_fn(batch, args.max_seq_len),
        )
    else:
        dataset = [tokenize_example(example, tokenizer) for example in dataset]
        print(len(dataset))
        dataset = [example for example in dataset if len(example['labels']) <= args.max_seq_len]
        print(len(dataset))
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, args.max_seq_len),
        )

    # Model
    model = MambaDecoderForCausalLM.from_pretrained(args.model_name, use_mambapy=True)
    model.eval()
    model.cuda()
    device = "cuda"

    if args.cache_dir:
        cache = TrainableMambaCache(config=model.config)
        state_dict = torch.load(os.path.join(args.cache_dir, 'training_state.bin'))
        cache.load_state_dict(state_dict["model_state_dict"])
        encoder_cache_params = MambaCache(
            config=model.config, max_batch_size=1, dtype=model.dtype, device = device
        )
        encoder_cache_params.conv_states = cache.learned_conv_state.expand([-1, args.batch_size, -1, -1]).to(device)
        encoder_cache_params.ssm_states = cache.learned_ssm_state.expand([-1, args.batch_size, -1, -1]).to(device)
    
    '''
    for idx, batch in enumerate(data_loader):
        print(idx)
        labels=batch["labels"]
        flat_labels = labels.view(-1)
        flat_labels = flat_labels[flat_labels != -100]
        print('Flat Labels: ', flat_labels)
    '''

    metric=0
    total=0
    for idx, batch in enumerate(data_loader):
        print(idx)
        batch = {k: v.to(device) for k, v in batch.items()}
        if batch["labels"].shape[0] == args.batch_size:
            if args.metric == 'contrastive-accuracy':
                with torch.no_grad():
                    if args._get_args:
                        outputs1= model(input_ids=batch["sol1"], labels=batch["sol1"], encoder_cache_params=encoder_cache_params)                        
                        outputs2= model(input_ids=batch["sol2"], labels=batch["sol2"], encoder_cache_params=encoder_cache_params)
                    else:
                        outputs1= model(input_ids=batch["sol1"], labels=batch["sol1"])                        
                        outputs2= model(input_ids=batch["sol2"], labels=batch["sol2"])
                loss1 = outputs1.loss
                loss2 = outputs2.loss
                print(loss1.item(),loss2.item(), batch["label"][0])
                if loss1.item() < loss2.item():
                    pred = 17
                elif loss1.item() > loss2.item():
                    pred = 18
                
                if pred == batch["label"][0]:
                    metric+=1
                print('Correct:', metric)
                total+=1
            else:
                with torch.no_grad():
                    if args.metric  == "loss":
                        if args.cache_dir:
                            outputs = model(**batch, encoder_cache_params=encoder_cache_params)
                        else:
                            outputs = model(**batch)
                        batch_metric = outputs.loss
                        print('Batch Loss:', batch_metric.item())
                        if torch.isnan(batch_metric):
                            print(batch['labels'])
                        else:
                            metric+=batch_metric.item()
                            total+=1
                    elif args.metric == "accuracy":
                        outputs = model(batch['input_ids'])
                        outputs = model(batch['input_ids'])
                        logits = outputs.logits[:, -1, :]
                        batch_correct=calculate_correct(logits, batch['labels'])
                        metric += batch_correct
                        total += batch["labels"].shape[0]

                                    
    print('Metric:', metric/total)
    if args.output_path:
        with open(args.output_path, "w") as file:
            file.write(f"{metric}")


if __name__ == "__main__":
    main()
