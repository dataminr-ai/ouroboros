import json

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers.cache_utils import MambaCache

import ouroboros.encode_dataset as ed
from ouroboros.decoder_utils import reconstruct


def load_dataset(file_path):
    dataset = []
    with open(file_path, "r") as file:
        for line in file:
            dataset.append(json.loads(line))
    return dataset


def tokenize_example(example, tokenizer, add_eos=False):
    tokenized_inputs = tokenizer(
        example["inputs"],
        return_attention_mask=False,
    )
    tokenized_label = tokenizer(
        example["label_str"],
        return_attention_mask=False,
    )
    if add_eos:
        tokenized_label["input_ids"] += [tokenizer.eos_token_id]
    input_ids = tokenized_inputs["input_ids"] + tokenized_label["input_ids"]
    labels = [-100] * len(tokenized_inputs["input_ids"]) + tokenized_label["input_ids"]
    return {"input_ids": input_ids, "labels": labels}

def classification_loss_outputs(batch, model, encoder_cache_params, args, config=None, tokenizer=None, decoder=None):
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
            decoded_cache = reconstruct(decoder, tokenizer, learned_cache_params).to(
                args.device
            )
            with torch.no_grad():
                recon_cache_params = ed.get_cache_params(decoded_cache, model)

            # define distance function
            ssm_dist = torch.norm(
                learned_cache_params.ssm_states - recon_cache_params.ssm_states
            )
            conv_dist = torch.norm(
                learned_cache_params.conv_states - recon_cache_params.conv_states
            )

            # Loss
            loss = outputs.loss + args.reg_strength * (ssm_dist + conv_dist)
        return loss, outputs

def contrastive_tokenize_example(example, tokenizer):
    positive_input_ids = tokenizer(
        example["positive"],
        return_attention_mask=False,
    )["input_ids"]
    negative_input_ids = tokenizer(
        example["negative"],
        return_attention_mask=False,
    )["input_ids"]
    return {
        "positive_input_ids": positive_input_ids,
        "negative_input_ids": negative_input_ids,
    }


def collate_fn(x, max_len=200):
    # NOTE(rlogan): This is slow but correct
    # TODO: Make the max size configurable instead of 128
    if max_len:
        max_seq_len = min(max(len(x_["labels"]) for x_ in x), max_len)
    else:
        max_seq_len = max(len(x_["labels"]) for x_ in x)
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


def contrastive_collate_fn(x, max_len=200):
    if max_len:
        max_positive_len = min(max(len(x_["positive_input_ids"]) for x_ in x), max_len)
        max_negative_len = min(max(len(x_["negative_input_ids"]) for x_ in x), max_len)
    else:
        max_positive_len = max(len(x_["positive_input_ids"]) for x_ in x)
        max_negative_len = max(len(x_["negative_input_ids"]) for x_ in x)
    batch_size = len(x)

    positive_input_ids = torch.zeros((batch_size, max_positive_len), dtype=torch.int64)
    for i, x_ in enumerate(x):
        positive_input_ids[i, : len(x_["positive_input_ids"])] = torch.tensor(
            x_["positive_input_ids"]
        )[:max_positive_len]
    negative_input_ids = torch.zeros((batch_size, max_negative_len), dtype=torch.int64)
    for i, x_ in enumerate(x):
        negative_input_ids[i, : len(x_["negative_input_ids"])] = torch.tensor(
            x_["negative_input_ids"]
        )[:max_negative_len]

    return {
        "positive_input_ids": positive_input_ids,
        "negative_input_ids": negative_input_ids,
    }

def seqlen_normalize_logp(labels, logp):
    valid_token_mask = (labels[..., 1:] != -100).float()
    summed = (logp * valid_token_mask).sum(dim=-1)
    avg = summed / valid_token_mask.sum(dim=-1)
    return avg

def contrastive_logp(batch, model, encoder_cache_params=None):
    positive_input_ids = batch["positive_input_ids"]
    positive_labels = positive_input_ids.clone()
    positive_labels[positive_labels == 0] = -100
    positive_outputs = model(
        batch["positive_input_ids"],
        encoder_cache_params=encoder_cache_params,
    )
    positive_logp = (
        -F.cross_entropy(
            positive_outputs.logits[..., :-1, :]
            .contiguous()
            .view(-1, positive_outputs.logits.size(-1)),
            positive_labels[..., 1:].contiguous().view(-1),
            reduction="none",
        )
        .view(positive_input_ids.size(0), -1)
    ) 
    norm_positive_logp = seqlen_normalize_logp(positive_labels, positive_logp)

    negative_input_ids = batch["negative_input_ids"]
    negative_labels = negative_input_ids.clone()
    negative_labels[negative_labels == 0] = -100
    negative_outputs = model(
        batch["negative_input_ids"],
        encoder_cache_params=encoder_cache_params,
    )
    negative_logp = (
        -F.cross_entropy(
            negative_outputs.logits[..., :-1, :]
            .contiguous()
            .view(-1, negative_outputs.logits.size(-1)),
            negative_labels[..., 1:].contiguous().view(-1),
            reduction="none",
        )
        .view(negative_input_ids.size(0), -1)
    )

    norm_negative_logp = seqlen_normalize_logp(negative_labels, negative_logp)    

    #Ref
    positive_ref_outputs = model(
            batch["positive_input_ids"],
        )
    positive_ref_logp = (
            -F.cross_entropy(
                positive_ref_outputs.logits[..., :-1, :]
                .contiguous()
                .view(-1, positive_outputs.logits.size(-1)),
                positive_labels[..., 1:].contiguous().view(-1),
                reduction="none",
            )
            .view(positive_input_ids.size(0), -1)
        )
    norm_positive_ref_logp = seqlen_normalize_logp(positive_labels, positive_ref_logp)

    negative_ref_outputs = model(
            batch["negative_input_ids"],
        )
    negative_ref_logp = (
            -F.cross_entropy(
                negative_ref_outputs.logits[..., :-1, :]
                .contiguous()
                .view(-1, negative_outputs.logits.size(-1)),
                negative_labels[..., 1:].contiguous().view(-1),
                reduction="none",
            )
            .view(negative_input_ids.size(0), -1)
        )
    norm_negative_ref_logp = seqlen_normalize_logp(negative_labels, negative_ref_logp)

    return norm_positive_logp, norm_negative_logp, norm_positive_ref_logp, norm_negative_ref_logp

def contrastive_accuracy_loss(batch, model, args, encoder_cache_params=None, config=None, tokenizer=None, decoder=None):
    dpo_weight = args.dpo_weight
    positive_logp, negative_logp, positive_ref_logp, negative_ref_logp= contrastive_logp(batch, model, encoder_cache_params)
    #accuracy
    diff = positive_logp - negative_logp
    preds = diff > 0
    acc_num = preds.sum().item()
    acc_denom = preds.size(0)
    #loss
    inner = dpo_weight * (
        positive_logp - positive_ref_logp - negative_logp + negative_ref_logp
    )
    loss1 = -F.logsigmoid(inner).mean()

    if not args.reg:
        loss = loss1
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
        decoded_cache = reconstruct(decoder, tokenizer, learned_cache_params).to(
            args.device
        )
        model.eval()
        with torch.no_grad():
            recon_cache_params = ed.get_cache_params(decoded_cache, model)
        model.train()
        # define distance function
        ssm_dist = torch.norm(
            learned_cache_params.ssm_states - recon_cache_params.ssm_states
        )
        conv_dist = torch.norm(
            learned_cache_params.conv_states - recon_cache_params.conv_states
        )

        # Loss
        loss = loss1 + args.reg_strength * (ssm_dist + conv_dist)

    return acc_num, acc_denom, loss

def validate_contrastive(data_loader, model,args, encoder_cache_params=None, config=None, tokenizer=None, decoder=None ):
    model.eval()
    acc_num = 0
    acc_denom = 0
    loss = 0
    total = 0
    steps = 0
    max_steps = len(data_loader)

    with torch.no_grad():
        with tqdm(total=max_steps, desc="Validation Progress") as pbar:
            pbar.update(steps)
            for i, batch in enumerate(data_loader):
                if i == args.validation_limit:
                    break
                batch = {k: v.to(args.device) for k, v in batch.items()}
                batch_size = next(iter(batch.values())).size(0)
                if encoder_cache_params:
                    encoder_cache_params.resize(batch_size)
                num, denom, batch_loss = contrastive_accuracy_loss(batch, model, args, encoder_cache_params, config, tokenizer, decoder)
                acc_num += num
                acc_denom += denom
                loss += batch_loss.item()
                total += 1  # This should probably be normalized by batch size
                steps += 1
                pbar.update(1)

    valid_loss = loss / total
    valid_acc = acc_num / (acc_denom + 1e-13)
    return valid_loss, valid_acc

def validate_classification(data_loader, model, args, encoder_cache_params=None, config=None, tokenizer=None, decoder=None):
    model.eval()
    acc_num = 0
    acc_denom = 0
    loss = 0
    total = 0
    steps = 0
    max_steps = len(data_loader)

    with torch.no_grad():
        with tqdm(total=max_steps, desc="Validation Progress") as pbar:
            pbar.update(steps)
            for i, batch in enumerate(data_loader):
                if i == args.validation_limit:
                    break
                batch = {k: v.to(args.device) for k, v in batch.items()}
                batch_size = next(iter(batch.values())).size(0)
                if encoder_cache_params:
                    encoder_cache_params.resize(batch_size)
                batch_loss, outputs = classification_loss_outputs(batch, model, encoder_cache_params, args, config=None, tokenizer=None, decoder=None)
                labels = batch["labels"][:, 1:]
                preds = outputs.logits.argmax(dim=-1)[:, :-1]
                # Next, we need to ignore all of the non-label token predictions
                preds[labels == -100] = -100
                # Finally, the accuracy is where the pred-label equal token count equals the sequence length.
                acc_num += (
                    ((preds == labels).sum(dim=-1) == labels.size(1))
                    .sum()
                    .item()
                )
                acc_denom += labels.size(0)
                loss += batch_loss.item()
                total += 1  # This should probably be normalized by batch size
                steps += 1
                pbar.update(1)
    valid_loss = loss / total
    valid_acc = acc_num / (acc_denom + 1e-13)
    return valid_loss, valid_acc

