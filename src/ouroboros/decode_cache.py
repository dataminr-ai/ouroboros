import argparse

import torch
from transformers import AutoTokenizer
from transformers.cache_utils import MambaCache

from ouroboros.decoder_utils import reconstruct_text
from ouroboros.models import (
    MambaDecoderForCausalLM,
    TrainableMambaCache,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--decoder",
        type=str,
        required=True,
        help="Path to the pretrained decoder model",
    )
    parser.add_argument(
        "--learned_cache",
        type=str,
        required=True,
        help="Path to the learned cache state_dict file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        default=None,
        help="Path to the learned cache state_dict file",
    )

    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer")

    parser.add_argument(
        "--max_seq_len",
        type=int,
        required=False,
        default=100,
        help="Maximum tokens to reconstruct",
    )

    parser.add_argument(
        "--validation_limit",
        type=int,
        default=-1,
        help="Limits the number of validation batches (for development)",
    )

    args = parser.parse_args()

    # Reconstructor
    decoder = MambaDecoderForCausalLM.from_pretrained(
        args.decoder, torch_dtype=torch.float16
    )
    decoder.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Cache
    cache = TrainableMambaCache(config=decoder.config)

    state_dict = torch.load(args.learned_cache)
    cache.load_state_dict(state_dict["model_state_dict"])
    cache_params = MambaCache(
        config=decoder.config, max_batch_size=1, dtype=decoder.dtype
    )
    cache_params.conv_states = cache.learned_conv_state
    cache_params.ssm_states = cache.learned_ssm_state

    # Add to conv cache
    recons = reconstruct_text(decoder, tokenizer, cache_params, args.max_seq_len)
    print(recons)

    if args.output_dir:
        with open(f"{args.output_dir}/prompt.txt", "w") as f:
            for recon in recons:
                f.write(recon + "\n")


if __name__ == "__main__":
    main()
