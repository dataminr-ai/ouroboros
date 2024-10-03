import argparse

import torch
from transformers import AutoTokenizer
from transformers.cache_utils import MambaCache

from ouroboros.models import (
    MambaDecoderForCausalLM,
    TrainableMambaCache,
)


def reconstruct(model, tokenizer, cache_params, max_seq_len=1000):
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
    # while not finished.all():  # Continue until all sequences have an EOS token
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
    return generated


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
        default=1000,
        help="Maximum tokens to reconstruct",
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

    recons_tokens = reconstruct(decoder, tokenizer, cache_params, args.max_seq_len)
    recons = tokenizer.batch_decode(recons_tokens, skip_special_tokens=True)
    print(recons)

    if args.output_dir:
        with open(f"{args.output_dir}/prompt.txt", "w") as f:
            for recon in recons:
                f.write(recon + "\n")


if __name__ == "__main__":
    main()
