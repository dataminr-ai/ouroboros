from os import PathLike
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedModel
from transformers.models.mamba.modeling_mamba import MambaCache


def save_checkpoint(model: PreTrainedModel, optimizer: Optimizer, scheduler: LRScheduler, epoch: int, step: int, checkpoint_path: PathLike):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        save_directory=str(checkpoint_path),
    )
    state_dict_path = checkpoint_path / "training_state.pt"
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(checkpoint, str(state_dict_path))


def load_checkpoint(checkpoint_path: PathLike, ModelClass: Type[PreTrainedModel]) -> Tuple[PreTrainedModel, Dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No directory found at [{str(checkpoint_path)}]")
    
    if not checkpoint_path.is_dir():
        raise NotADirectoryError(f"Path[{str(checkpoint_path)}] exists but is not a directory")
    
    model = ModelClass.from_pretrained(pretrained_model_name_or_path=str(checkpoint_path))
    state_dicts = {}

    if (checkpoint_path / "training_state.pt").exists():
        state_dicts = torch.load(str(checkpoint_path / "training_state.pt"))

    if (checkpoint_path / "training_state.bin").exists():
        state_dicts = torch.load(str(checkpoint_path / "training_state.bin"))

    return model, state_dicts


def get_cache_state_for_batch(batch: torch.Tensor, model: PreTrainedModel) -> torch.Tensor:
    outputs = model(batch, output_hidden_states=True, use_cache=True, return_dict=True)
    hidden_states = outputs.cache_params
    return hidden_states


def move_cache_to_device(mamba_cache: MambaCache, device: torch.device) -> MambaCache:
    """
    Function to move MambaCache parameters to a device. 
    The MambaCache class in huggingface does not have a .to(device) function
        and needs to be handled this way.
    """
    for key in mamba_cache.ssm_states:
        mamba_cache.ssm_states[key] = mamba_cache.ssm_states[key].to(device)
    for key in mamba_cache.conv_states:
        mamba_cache.conv_states[key] = mamba_cache.conv_states[key].to(device)

    return mamba_cache