import numpy as np
import torch
import piq
from skimage.metrics import structural_similarity as ssim_metric

def compute_psnr(gt: np.ndarray, pred: np.ndarray, max_val: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        gt: Ground truth image/array.
        pred: Predicted image/array.
        max_val: Maximum possible pixel value of the images (default 1.0 for normalized arrays).
        
    Returns:
        PSNR value in dB. Returns infinity if MSE is exactly zero.
    """
    mse = np.mean((gt - pred) ** 2)
    if mse == 0:
        return float('inf')
    return float(20 * np.log10(max_val) - 10 * np.log10(mse))

def compute_ssim(gt: np.ndarray, pred: np.ndarray, max_val: float = 1.0) -> float:
    """
    Compute Structural Similarity Index (SSIM).
    
    Args:
        gt: Ground truth image/array.
        pred: Predicted image/array.
        max_val: Data range of the input image.
        
    Returns:
        SSIM index.
    """
    # Ensure inputs are appropriate precision, cast to float64 to avoid precision loss
    gt = gt.astype(np.float64)
    pred = pred.astype(np.float64)
    return float(ssim_metric(gt, pred, data_range=max_val))

def compute_mae(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error (MAE).
    
    Args:
        gt: Ground truth image/array.
        pred: Predicted image/array.
        
    Returns:
        MAE value.
    """
    return float(np.mean(np.abs(gt - pred)))

def compute_mse(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Compute Mean Squared Error (MSE).
    
    Args:
        gt: Ground truth image/array.
        pred: Predicted image/array.
        
    Returns:
        MSE value.
    """
    return float(np.mean((gt - pred) ** 2))

def compute_fsim(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Compute Feature Similarity Index Measure (FSIM) for grayscale images.
    
    Args:
        gt: Ground truth image/array.
        pred: Predicted image/array.
        
    Returns:
        FSIM value.
    """
    gt_tensor = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0)
    pred_tensor = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        score = piq.fsim(pred_tensor, gt_tensor, data_range=1.0, chromatic=False)
    return float(score.item())

