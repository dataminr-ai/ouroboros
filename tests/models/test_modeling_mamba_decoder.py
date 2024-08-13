import tempfile

import torch
from transformers.cache_utils import MambaCache
from transformers.models.mamba import MambaConfig, MambaForCausalLM

from ouroboros.models.configuration_mamba_decoder import MambaDecoderConfig
from ouroboros.models.modeling_mamba_decoder import MambaDecoderForCausalLM


def test_mixer_changes():
    # Randomly initialize a model
    config = MambaDecoderConfig(use_mambapy=True)
    model = MambaDecoderForCausalLM(config)
    model.train()

    # Create a random input
    batch_size = 2
    sequence_length = 32
    input_ids = torch.randint(0, config.vocab_size, size=(batch_size, sequence_length))

    # Instantiate a cache
    encoder_output = model(input_ids, use_cache=True)
    encoder_cache_params = encoder_output["cache_params"]

    # In theory, if we feed cache_params with zeroed out hidden state our output
    # should be identical to above.
    encoder_cache_params.ssm_states.zero_()
    decoder_output = model(input_ids, encoder_cache_params=encoder_cache_params)

    print(encoder_output.logits)
    print(decoder_output.logits)
    assert torch.allclose(encoder_output.logits, decoder_output.logits, atol=1e-3)


def test_mambapy():
    encoder = MambaForCausalLM(config=MambaConfig())

    config = MambaDecoderConfig(use_mambapy=True)
    model_w_mambapy = MambaDecoderForCausalLM(config)
    model_w_mambapy.train()
    with tempfile.TemporaryDirectory() as tempdir:
        model_w_mambapy.save_pretrained(tempdir)
        model_wout_mambapy = MambaDecoderForCausalLM.from_pretrained(tempdir, use_mambapy=False)

    # Create a random input
    batch_size = 2
    sequence_length = 32
    input_ids = torch.randint(0, config.vocab_size, size=(batch_size, sequence_length))

    # Instantiate a cache
    encoder_output = encoder(input_ids, use_cache=True)
    encoder_cache_params = encoder_output["cache_params"]

    print('w/out Mambapy')
    output_wout_mambapy = model_wout_mambapy(input_ids, use_cache=False, encoder_cache_params=encoder_cache_params)

    print('w/ Mambapy')
    output_w_mambapy = model_w_mambapy(input_ids, use_cache=False, encoder_cache_params=encoder_cache_params)

    assert torch.allclose(output_w_mambapy.logits, output_wout_mambapy.logits, atol=1e-3)
