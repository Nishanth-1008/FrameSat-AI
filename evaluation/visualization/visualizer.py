import os
import matplotlib.pyplot as plt
import numpy as np

def save_comparison_figure(
    t0: np.ndarray, 
    gt: np.ndarray, 
    pred: np.ndarray, 
    out_path: str, 
    title: str = "Evaluation Comparison"
):
    """
    Save a 4-panel side-by-side comparison figure.
    
    Panels:
    1. Input Frame (t0)
    2. Ground Truth Frame (t1)
    3. Prediction (pred)
    4. Absolute Error Heatmap (|gt - pred|)
    
    Args:
        t0: Input frame array.
        gt: Ground truth target frame array.
        pred: Predicted target frame array.
        out_path: File path to save the generated figure.
        title: Title to display at the top of the figure.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. Input t0
    axes[0].imshow(t0, cmap='gray')
    axes[0].set_title("Input (t0)")
    axes[0].axis('off')
    
    # 2. Ground Truth t1
    axes[1].imshow(gt, cmap='gray')
    axes[1].set_title("Ground Truth (t1)")
    axes[1].axis('off')
    
    # 3. Prediction
    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title("Prediction")
    axes[2].axis('off')
    
    # 4. Difference Heatmap
    diff = np.abs(gt - pred)
    # Using vmin=0 and vmax=0.2 typical for normalized differences in SEVIR
    im = axes[3].imshow(diff, cmap='hot', vmin=0.0, vmax=0.2)
    axes[3].set_title("Absolute Error Heatmap")
    axes[3].axis('off')
    
    # Add colorbar for the heatmap
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    
    # Add main title
    plt.suptitle(title, fontsize=14)
    
    # Save and close
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
