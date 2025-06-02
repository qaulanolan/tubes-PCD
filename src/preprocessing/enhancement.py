# src/preprocessing/enhancement.py
"""
Module for image enhancement techniques in image processing.
"""

import cv2
import numpy as np

def apply_histogram_equalization(image):
    """
    Apply histogram equalization to enhance image contrast.
    
    Parameters:
    - image: Input image (numpy array)
    
    Returns:
    - Enhanced image
    """
    if len(image.shape) == 3:  # Color image
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # Apply histogram equalization to value channel
        hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
        # Convert back to RGB
        enhanced_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    else:  # Grayscale image
        enhanced_image = cv2.equalizeHist(image)
    
    return enhanced_image

# def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
#     """
#     Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
#     Parameters:
#     - image: Input image (numpy array)
#     - clip_limit: Threshold for contrast limiting
#     - tile_grid_size: Size of grid for histogram equalization
    
#     Returns:
#     - Enhanced image
#     """
#     # Create CLAHE object
#     clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
#     if len(image.shape) == 3:  # Color image
#         # Convert to LAB color space
#         lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
#         # Apply CLAHE to L channel
#         lab[:,:,0] = clahe.apply(lab[:,:,0])
#         # Convert back to RGB
#         enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
#     else:  # Grayscale image
#         enhanced_image = clahe.apply(image)
    
#     return enhanced_image

def apply_contrast_stretching(image, min_percentile=2, max_percentile=98):
    """
    Apply contrast stretching using percentile-based normalization.
    
    Parameters:
    - image: Input image (numpy array)
    - min_percentile: Lower percentile for normalization
    - max_percentile: Upper percentile for normalization
    
    Returns:
    - Enhanced image
    """
    if len(image.shape) == 3:  # Color image
        enhanced_image = np.zeros_like(image, dtype=np.uint8)
        
        # Process each channel separately
        for i in range(3):
            channel = image[:,:,i]
            
            # Get percentiles
            min_val = np.percentile(channel, min_percentile)
            max_val = np.percentile(channel, max_percentile)
            
            # Clip the values to the computed min and max
            channel_stretched = np.clip(channel, min_val, max_val)
            
            # Scale to full range [0, 255]
            if max_val > min_val:
                channel_stretched = ((channel_stretched - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            
            enhanced_image[:,:,i] = channel_stretched
    else:  # Grayscale image
        # Get percentiles
        min_val = np.percentile(image, min_percentile)
        max_val = np.percentile(image, max_percentile)
        
        # Clip and scale
        enhanced_image = np.clip(image, min_val, max_val)
        if max_val > min_val:
            enhanced_image = ((enhanced_image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    return enhanced_image

def apply_gamma_correction(image, gamma=1.0):
    """
    Apply gamma correction to adjust image brightness/contrast.
    
    Parameters:
    - image: Input image (numpy array)
    - gamma: Gamma value (< 1: brighter, > 1: darker)
    
    Returns:
    - Enhanced image
    """
    # Normalize to [0, 1]
    normalized = image.astype(np.float32) / 255.0
    
    # Apply gamma correction
    corrected = np.power(normalized, gamma)
    
    # Scale back to [0, 255]
    enhanced_image = (corrected * 255).astype(np.uint8)
    
    return enhanced_image

def apply_unsharp_mask(image, kernel_size=5, sigma=1.0, amount=1.0, threshold=0):
    """
    Apply unsharp masking to sharpen the image.
    
    Parameters:
    - image: Input image (numpy array)
    - kernel_size: Size of Gaussian kernel
    - sigma: Standard deviation of Gaussian
    - amount: Weight of the sharpening effect
    - threshold: Minimum brightness difference to apply sharpening
    
    Returns:
    - Sharpened image
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create blurred version of the image
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    # Calculate unsharp mask
    unsharp_mask = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    
    # Apply threshold if specified
    if threshold > 0:
        # Calculate absolute difference
        diff = cv2.absdiff(image, blurred)
        # Create mask where difference exceeds threshold
        mask = diff > threshold
        # Apply unsharp mask only where the mask is True
        sharpened = np.where(mask, unsharp_mask, image)
    else:
        sharpened = unsharp_mask
    
    return sharpened.astype(np.uint8)