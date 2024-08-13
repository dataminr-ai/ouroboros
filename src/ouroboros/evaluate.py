import datasets
import pickle
from transformers import AutoConfig, MambaForCausalLM, AutoTokenizer, AutoModelForCausalLM
import torch
import os
import gc
import json
import re
import argparse

import ouroboros.encode_dataset as ed
from ouroboros.models import MambaDecoderForCausalLM, MambaDecoderConfig

AutoModelForCausalLM.register(MambaDecoderConfig, MambaDecoderForCausalLM)

def reconstruct(model, tokenizer, cache, chunk_size, batch_size):
   cache.seqlen_offset = 0
   preds = []
   for idx in range(chunk_size + 1):
       #print(idx)
       if idx == 0:
           inputs = torch.zeros((cache.conv_states[0].shape[0], 1)).int().to("cuda")
           cache_params = cache
       else:
           inputs = next_tokens
       with torch.no_grad():
           cache_position = torch.tensor([0,1,2,3,4])
           outputs = model(
               input_ids=inputs,
               cache_params=cache_params,
               use_cache=True,
               cache_position=cache_position,
               output_dict=True,
               use_mambapy=True,
               forward_type="fast"
           )
       cache_params = outputs.cache_params
       next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1).view(-1, 1)
       preds.append(next_tokens.to("cpu"))
   gen = torch.cat(preds, dim=1).to("cpu")
   recons = []
   for i in range(gen.shape[0]):
       recons.append(tokenizer.decode(gen[i]))
   del inputs, cache, cache_params, outputs, next_tokens, gen, preds
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
    metric = datasets.load_metric("rouge")  # Load metric

    tokenizer = AutoTokenizer.from_pretrained(base_model)  # Load Tokenizer

    # Load Encoder
    encoder = MambaForCausalLM.from_pretrained(
        base_model,
    )
    encoder.eval()
    encoder.cuda()

    # Load Decoder
    model = AutoModelForCausalLM.from_pretrained(ckpt_path, use_mambapy=True)
    midx = extract_step_number(ckpt_path)
    model.eval()
    model.cuda()

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
        recons = reconstruct(model, tokenizer, cache_params, chunk_size, batch_size)
        reconstructed.append(recons)
        del batch, recons
        torch.cuda.empty_cache()
        gc.collect()
        print(torch.cuda.memory_allocated())

    output_file = os.path.join(output_dir, str(midx) + ".pkl")
    with open(output_file, "wb") as f:
        pickle.dump(reconstructed, f)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Score
    for idx, batch in enumerate(batched_chunks):
        reference_text = tokenizer.batch_decode(
            batch["input_ids"], skip_special_tokens=True
        )
        reconstructed_text = reconstructed[idx]
        metric.add(predictions=[reconstructed_text], references=[reference_text])

    score = metric.compute()

    json_data = json.dumps(score, indent=4)
    output_file = os.path.join(output_dir, str(midx) + ".json")
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
