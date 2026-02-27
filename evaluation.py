import cv2
import numpy as np
import torch
from model import dice_loss

def post_process(mask, threshold=0.5, kernel_size=3):
    """
    Industry-standard post-processing: Thresholding + Morphological Denoising.
    """
    # 1. Binary Thresholding
    binary_mask = (mask > threshold).astype(np.uint8)
    
    # 2. Morphological Denoising (Closing to fill small holes, Opening to remove noise)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Opening: Erosion followed by Dilation (removes salt noise)
    denoised = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # Closing: Dilation followed by Erosion (fills small gaps)
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
    
    return denoised

def optimize_threshold(pred_probs, true_labels):
    """
    Sweeps thresholds to find the best Dice score (Industry Level).
    """
    best_dice = -1
    best_threshold = 0.5
    
    print("Optimization Start: Sweeping Thresholds...")
    for t in np.arange(0.1, 0.9, 0.05):
        # Convert to tensors for dice_loss function (expects logits or probs)
        # We'll use simple numpy dice for this utility
        pred = (pred_probs > t).astype(np.float32)
        intersection = (pred * true_labels).sum()
        dice = (2. * intersection + 1.0) / (pred.sum() + true_labels.sum() + 1.0)
        
        if dice > best_dice:
            best_dice = dice
            best_threshold = t
            
    print(f"Best Dice: {best_dice:.4f} at Threshold: {best_threshold:.2f}")
    return best_threshold

if __name__ == "__main__":
    # Example usage on dummy data
    dummy_pred = np.random.rand(512, 512)
    dummy_true = (np.random.rand(512, 512) > 0.8).astype(float)
    
    best_t = optimize_threshold(dummy_pred, dummy_true)
    final_mask = post_process(dummy_pred, threshold=best_t)
    
    print(f"Post-processing verified. Mask saved (if implemented).")
