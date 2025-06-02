# src/preprocessing/noise_reduction.py
"""
Module for noise reduction techniques in image processing.
"""

import cv2
import numpy as np

def apply_gaussian_filter(image, kernel_size=5, sigma=1.0):
    """
    Apply Gaussian filter for noise reduction.
    
    Parameters:
    - image: Input image (numpy array)
    - kernel_size: Size of Gaussian kernel (must be odd)
    - sigma: Standard deviation of the Gaussian
    
    Returns:
    - Filtered image
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Apply Gaussian blur
    if len(image.shape) == 3:  # Color image
        filtered_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    else:  # Grayscale image
        filtered_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    return filtered_image

def apply_median_filter(image, kernel_size=5):
    """
    Apply median filter for noise reduction.
    Effective for salt and pepper noise.
    
    Parameters:
    - image: Input image (numpy array)
    - kernel_size: Size of median kernel (must be odd)
    
    Returns:
    - Filtered image
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Apply median blur
    filtered_image = cv2.medianBlur(image, kernel_size)
    
    return filtered_image

def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral filter for edge-preserving noise reduction.
    
    Parameters:
    - image: Input image (numpy array)
    - d: Diameter of each pixel neighborhood
    - sigma_color: Filter sigma in color space
    - sigma_space: Filter sigma in coordinate space
    
    Returns:
    - Filtered image
    """
    filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    return filtered_image

def apply_nlm_filter(image, h=10, template_window_size=7, search_window_size=21):
    """
    Apply Non-Local Means filter for noise reduction.
    
    Parameters:
    - image: Input image (numpy array)
    - h: Filter strength parameter
    - template_window_size: Size of template patch
    - search_window_size: Size of search window
    
    Returns:
    - Filtered image
    """
    if len(image.shape) == 3:  # Color image
        filtered_image = cv2.fastNlMeansDenoisingColored(
            image, None, h, h, template_window_size, search_window_size)
    else:  # Grayscale image
        filtered_image = cv2.fastNlMeansDenoising(
            image, None, h, template_window_size, search_window_size)
    
    return filtered_image