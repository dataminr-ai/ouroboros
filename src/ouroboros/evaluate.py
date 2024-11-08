import argparse
import json
import logging
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, MambaForCausalLM

import ouroboros.encode_dataset as ed
from ouroboros.decoder_utils import (
    extract_step_number,
    score_dataset,
)
from ouroboros.models import MambaDecoderConfig, MambaDecoderForCausalLM


AutoModelForCausalLM.register(MambaDecoderConfig, MambaDecoderForCausalLM)


def main(
    base_model, eval_file, chunk_size, batch_size, output_dir, ckpt_path
):
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

    # Compute metric
    score, comparison  = score_dataset(model, tokenizer, encoder, tokenized_dataset, chunk_size, batch_size)
    
    #Write output
    comparison_json = json.dumps(comparison, indent=4)
    output_file = os.path.join(output_dir, str(midx) + "_decoded.json")
    with open(output_file, "w") as file:
        file.write(comparison_json)

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
