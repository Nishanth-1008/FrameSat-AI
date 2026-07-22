import os
import torch
from typing import Dict, Any, Tuple, Optional

def load_checkpoint(checkpoint_path: str, device: str = "cpu") -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Loads a PyTorch checkpoint and extracts the state dict.
    
    Supports both raw state_dicts (e.g. from RIFE pretrained weights)
    and wrapped state dicts (e.g. from our training loop with 'state_dict', 'epoch', etc.)
    
    Args:
        checkpoint_path (str): The absolute path to the .pth file.
        device (str): Device to map the tensors to.
        
    Returns:
        tuple: (state_dict, full_state)
               state_dict is the actual model weights.
               full_state is the complete checkpoint dictionary (or None if it was raw weights).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(state, dict) and 'state_dict' in state:
        return state['state_dict'], state
    
    return state, None

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_psnr: float, config: dict, path: str):
    """
    Standardizes checkpoint saving across the pipeline.
    """
    # Extract model state dict without DDP 'module.' prefix if present
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
        
    state = {
        'epoch': epoch,
        'state_dict': model_state,
        'optimizer': optimizer.state_dict(),
        'best_psnr': best_psnr,
        'config': config
    }
    
    torch.save(state, path)
