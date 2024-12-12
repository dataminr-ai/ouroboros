import gc
import re

import torch

import datasets
import ouroboros.encode_dataset as ed


def reconstruct(model, tokenizer, cache_params, max_seq_len=100):
    input_ids = torch.full(
        (cache_params.conv_states.size(1), 1),
        tokenizer.bos_token_id,
        device=cache_params.conv_states.device,
    )
    cache_position = torch.tensor([3], device=input_ids.device)
    generated = []
    eos_token_id = tokenizer.eos_token_id  # EOS token
    finished = torch.zeros(
        input_ids.size(0), dtype=torch.bool, device=input_ids.device
    )  # Track finished sequences
    end = False
    while not end:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                cache_params=cache_params,
                cache_position=cache_position,
                use_cache=True,
                output_dict=True,
            )
        input_ids = outputs.logits.argmax(dim=-1)
        cache_params = outputs.cache_params
        cache_position = cache_position[-1:] + 1
        generated.append(input_ids.to("cpu"))
        # Check for EOS token in each sequence
        finished |= (input_ids == eos_token_id).any(dim=-1)
        if finished.all() or len(generated) == max_seq_len:
            end = True
    generated = torch.cat(generated, dim=-1).to("cpu")
    del input_ids, cache_params, cache_position, outputs
    torch.cuda.empty_cache()
    gc.collect()
    return generated

def clean_eos(text_list):
    cleaned_list = [text.split("<|endoftext|>")[0] for text in text_list]
    return cleaned_list

def extract_step_number(path):
    match = re.search(r"step_(\d+)", path)
    if match:
        return int(match.group(1))
    else:
        return 0
    
def reconstruct_text(decoder, tokenizer, cache_params, max_seq_len=100):
    recons_tokens = reconstruct(decoder, tokenizer, cache_params, max_seq_len)
    recons1 = tokenizer.batch_decode(recons_tokens)
    recons = clean_eos(recons1)
    return recons

def score_dataset(model, tokenizer, encoder, tokenized_dataset, chunk_size, batch_size):
    metric = datasets.load_metric("rouge", trust_remote_code=True)
    chunked_dataset = ed.chunk_dataset(tokenized_dataset, chunk_size)
    batched_chunks = ed.batch_chunks(
        chunked_dataset, batch_size
    )

    # Reconstruct text using decoder
    reconstructed = []
    for idx, batch in enumerate(batched_chunks):
        print("Idx, ", idx)
        with torch.no_grad():
            input_ids = batch["input_ids"].to("cuda")
            cache_params = ed.get_cache_params(input_ids, encoder)
        recons=reconstruct_text(model, tokenizer, cache_params)
        reconstructed.append(recons)
        del batch, recons, cache_params
        torch.cuda.empty_cache()
        gc.collect()
        print(torch.cuda.memory_allocated())

    # Score
    comparison={'reference':[], 'reconstructed':[]}
    for idx, batch in enumerate(batched_chunks):
        reference_text = tokenizer.batch_decode(
            batch["input_ids"], skip_special_tokens=True
        )
        reconstructed_text = reconstructed[idx]
        comparison['reference'].extend(reference_text)
        comparison['reconstructed'].extend(reconstructed_text)
    metric.add_batch(predictions=comparison['reconstructed'], references=comparison['reference'])

    score = metric.compute()
    return score, comparison