from typing import Optional, Union

import torch
import torch.nn as nn
from transformers.models.mamba import MambaConfig


class TrainableMambaCache(nn.Module):
    def __init__(
        self,
        config: MambaConfig,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float16,
        device: Optional[Union[torch.device, str]] = None,
        learned_conv_state: Optional[torch.Tensor] = None,
        learned_ssm_state: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.dtype = dtype
        self.batch_size = batch_size
        self.intermediate_size = config.intermediate_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel

        if learned_conv_state is not None:
            self.learned_conv_state = torch.nn.Parameter(learned_conv_state)
        else:
            self.learned_conv_state = torch.nn.Parameter(
                torch.zeros(
                    config.num_hidden_layers,
                    1,
                    config.intermediate_size,
                    config.conv_kernel,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )

        if learned_ssm_state is not None:
            self.learned_ssm_state = torch.nn.Parameter(learned_ssm_state)
        else:
            self.learned_ssm_state = torch.nn.Parameter(
                torch.zeros(
                    config.num_hidden_layers,
                    1,
                    config.intermediate_size,
                    config.state_size,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )

        self.conv_states = self.learned_conv_state.expand([-1, batch_size, -1, -1])
        self.ssm_states = self.learned_ssm_state.expand([-1, batch_size, -1, -1])
