from os import PathLike
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedModel
from transformers.models.mamba.modeling_mamba import MambaCache


def save_checkpoint(model: Union[PreTrainedModel, torch.nn.Module], optimizer: Optimizer, scheduler: LRScheduler, step: int, checkpoint_path: PathLike, epoch: Optional[int] = None):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    if isinstance(model, PreTrainedModel):
        model.save_pretrained(
            save_directory=str(checkpoint_path),
        )
        checkpoint = {}
    else:
        checkpoint = {
            "model_state_dict": model.state_dict()
        }
    state_dict_path = checkpoint_path / "training_state.pt"
    checkpoint = {
        **checkpoint,
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
    encoder_cache_params = None
    if hasattr(outputs, "encoder_cache_params"):
        encoder_cache_params = outputs.encoder_cache_params
    cache_params = outputs.cache_params

    assert (encoder_cache_params is None) or (cache_params is None), "Cannot have both encoder_cache_params and cache_params at the same time"

    if encoder_cache_params is None:
        return cache_params

    if cache_params is None:
        return encoder_cache_params
        
    return None


def move_cache_to_device(mamba_cache: MambaCache, device: torch.device) -> MambaCache:
    """
    Function to move MambaCache parameters to a device. 
    The MambaCache class in huggingface modeling_mamba.py 
        does not have a .to(device) function 
        and needs to be handled this way.
    """
    for key in mamba_cache.ssm_states:
        mamba_cache.ssm_states[key] = mamba_cache.ssm_states[key].to(device)
    for key in mamba_cache.conv_states:
        mamba_cache.conv_states[key] = mamba_cache.conv_states[key].to(device)

    return mamba_cache