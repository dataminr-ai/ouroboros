import torch
import torch.nn as nn
from transformers.models.mamba import MambaConfig, MambaForCausalLM
from transformers.cache_utils import MambaCache
from ouroboros.models import MambaDecoderForCausalLM
from typing import Any, Dict, Optional, Tuple, Union

class TrainableMambaCache(nn.Module):
    def __init__(
            self,
            config: MambaConfig,
            prompt_cache: Optional[MambaCache] = None,
            batch_size: int = 1,
            dtype: torch.dtype = torch.float16,
            device: Optional[Union[torch.device, str]] = None,
        ): 
            super(TrainableMambaCache, self).__init__()

            self.dtype = dtype
            self.batch_size = batch_size
            self.intermediate_size = config.intermediate_size
            self.ssm_state_size = config.state_size
            self.conv_kernel_size = config.conv_kernel

            if prompt_cache:
                 self.conv_states = nn.Parameter(prompt_cache.conv_states)
                 self.ssm_states = nn.Parameter(prompt_cache.ssm_states)
            else:
                self.conv_states = nn.Parameter(torch.randn(
                    config.num_hidden_layers,
                    self.batch_size,
                    self.intermediate_size,
                    self.conv_kernel_size,
                    device=device,
                    dtype=dtype,
                ))
                self.ssm_states = nn.Parameter(torch.randn(
                    config.num_hidden_layers,
                    self.batch_size,
                    self.intermediate_size,
                    self.ssm_state_size,
                    device=device,
                    dtype=dtype,
                ))

class MambaCacheOptimizer(nn.Module):
    def __init__(self, 
                 model_name: str,
                 prompt_cache: Optional[MambaCache] = None
                 ):
        super(MambaCacheOptimizer, self).__init__()

        self.decoder = MambaDecoderForCausalLM.from_pretrained(model_name, use_mambapy=True)
        self.trainable_cache = TrainableMambaCache(config = self.decoder.config, prompt_cache=prompt_cache)
        
        # Freeze the decoder weights
        for param in self.decoder.parameters():
            param.requires_grad = False
            
    def forward(self, input_ids, labels, batch_size=1):
        
        stacked_conv_states = self.trainable_cache.conv_states.repeat(1, batch_size, 1, 1)
        stacked_ssm_states = self.trainable_cache.ssm_states.repeat(1, batch_size, 1, 1)
        stacked_cache = MambaCache(config = self.decoder.config, max_batch_size=batch_size)
        stacked_cache.conv_states = stacked_conv_states
        stacked_cache.ssm_states = stacked_ssm_states

        decoder_outputs = self.decoder(
                        input_ids=input_ids,
                        encoder_cache_params=stacked_cache,
                        return_dict=True,
                        labels=labels
        )
        
        return decoder_outputs

