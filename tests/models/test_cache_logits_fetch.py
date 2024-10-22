import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ouroboros.utils.model import get_cache_state_for_batch


def test_get_cache_params():
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
    prompt = ["Pick the best option that answers the question.\n"]
    token_prompt = tokenizer(prompt, return_tensors="pt")
    batch_size = token_prompt["input_ids"].shape[0]
    assert batch_size == len(prompt)

    with torch.no_grad():
        encoded_prompt = get_cache_state_for_batch(token_prompt["input_ids"], model)
    learned_conv_state = encoded_prompt.conv_states
    learned_ssm_state = encoded_prompt.ssm_states

    assert learned_conv_state is not None 
    assert learned_ssm_state is not None
    assert isinstance(learned_conv_state, torch.Tensor)
    assert isinstance(learned_ssm_state, torch.Tensor)
    assert list(learned_conv_state.shape) == [model.config.n_layer, batch_size, model.config.intermediate_size, model.config.conv_kernel]
    assert list(learned_ssm_state.shape) == [model.config.n_layer, batch_size, model.config.intermediate_size, model.config.state_size]

