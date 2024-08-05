import argparse
import transformers

from ouroboros.models.configuration_mamba_decoder import MambaDecoderConfig
from ouroboros.models.modeling_mamba_decoder import MambaDecoderForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    args = parser.parse_args()

    # Load the model
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.input)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.input)

    # Get the state_dict
    state_dict = model.state_dict()

    # Use to instantiate the decoder
    new_config = MambaDecoderConfig(**model.config.to_dict())
    new_model = MambaDecoderForCausalLM(config=new_config)
    new_model.load_state_dict(state_dict)

    # Write the output
    tokenizer.save_pretrained(args.output)
    new_model.save_pretrained(args.output)


if __name__ == '__main__':
    main()
