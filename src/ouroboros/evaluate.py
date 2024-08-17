import datasets
import logging
import pickle
from transformers import AutoConfig, MambaForCausalLM, AutoTokenizer, AutoModelForCausalLM
import torch
import os
import gc
import json
import re
import argparse
#from evaluate import load

import ouroboros.encode_dataset as ed
from ouroboros.models import MambaDecoderForCausalLM, MambaDecoderConfig

AutoModelForCausalLM.register(MambaDecoderConfig, MambaDecoderForCausalLM)

def reconstruct(model, tokenizer, cache_params, chunk_size):
    input_ids = torch.full((cache_params.conv_states.size(1), 1), tokenizer.bos_token_id, device=cache_params.conv_states.device)
    cache_position = torch.arange(0, model.config.conv_kernel, device=input_ids.device)
    generated = []
    for idx in range(chunk_size + 1):
        #print(idx)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                cache_params=cache_params,
                cache_position=cache_position,  # NOTE(rlogan): What next though?
                use_cache=True,
                output_dict=True,
            )
        input_ids = outputs.logits.argmax(dim=-1)
        cache_params = outputs.cache_params
        cache_position = cache_position[-1:] + 1
        generated.append(input_ids.to("cpu"))
    generated = torch.cat(generated, dim=-1).to("cpu")
    #print("Generated Ids: ", generated)
    recons = tokenizer.batch_decode(generated, skip_special_tokens=True)
    del input_ids, cache_params, cache_position, outputs, generated
    torch.cuda.empty_cache()
    gc.collect()
    return recons


def extract_step_number(path):
    match = re.search(r"step_(\d+)", path)
    if match:
        return int(match.group(1))
    else:
        return 0


def main(
    base_model, eval_file, chunk_size, batch_size, output_dir, ckpt_path
):
    metric = datasets.load_metric("rouge", trust_remote_code=True)  # Load metric

    tokenizer = AutoTokenizer.from_pretrained(base_model)  # Load Tokenizer

    # Load Encoder
    encoder = MambaForCausalLM.from_pretrained(
        base_model,
    )
    encoder.eval()
    encoder.cuda()

    # Load Decoder
    model = MambaDecoderForCausalLM.from_pretrained(ckpt_path)
    midx = extract_step_number(ckpt_path)
    model.eval()
    model.cuda()

    logging.basicConfig(level=logging.INFO)

    # Load Dataset
    raw_dataset = ed.read_dataset(eval_file)
    tokenized_dataset = ed.tokenize_dataset(raw_dataset, tokenizer)
    chunked_dataset = ed.chunk_dataset(tokenized_dataset, chunk_size)
    batched_chunks = ed.batch_chunks(
        chunked_dataset, batch_size
    )  # batched_chunks[batch_number]['input_ids'][instance_number]

    # Reconstruct text using decoder
    reconstructed = []
    for idx, batch in enumerate(batched_chunks):
        print("Idx, ", idx)
        with torch.no_grad():
            input_ids = batch["input_ids"].to("cuda")
            cache_params = ed.get_cache_params(input_ids, encoder)
        #print(f'Input ids: {input_ids}')
        recons = reconstruct(model, tokenizer, cache_params, chunk_size)
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
        metric.add(predictions=[reconstructed_text], references=[reference_text])

    #output_file = os.path.join(output_dir, str(midx) + ".pkl")
    #with open(output_file, "wb") as f:
     #   pickle.dump(comparison, f)
    comparison_json = json.dumps(comparison, indent=4)
    output_file = os.path.join(output_dir, str(midx) + "_decoded.json")
    with open(output_file, "w") as file:
        file.write(comparison_json)

    score = metric.compute()

    json_data = json.dumps(score, indent=4)
    output_file = os.path.join(output_dir, str(midx) + "_rouge.json")
    with open(output_file, "w") as file:
        file.write(json_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference script for a pre-trained model"
    )
    parser.add_argument(
        "--base_model", type=str, required=True, help="Base model name or path"
    )
    parser.add_argument(
        "--eval_file", type=str, required=True, help="Data for inference (.jsonl)"
    )
    parser.add_argument("--chunk_size", type=int, required=True, help="Sequence Length")
    parser.add_argument("--batch_size", type=int, required=True, help="Sequence Length")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the outputs"
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to the checkpoint file"
    )

    args = parser.parse_args()

    main(
        args.base_model,
        args.eval_file,
        args.chunk_size,
        args.batch_size,
        args.output_dir,
        args.ckpt_path,
    )
