# src/feature_extraction/color_features.py
"""
Module for extracting color-based features from images.
"""

import cv2
import numpy as np
from skimage import color

def extract_color_histogram(image, bins=32, normalize=True):
    """
    Extract color histogram features from an image.
    
    Parameters:
    - image: Input image (numpy array in RGB format)
    - bins: Number of bins per channel
    - normalize: Whether to normalize the histogram
    
    Returns:
    - Flattened histogram features
    """
    # Check if image is grayscale
    if len(image.shape) == 2 or image.shape[2] == 1:
        # Single channel histogram
        hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
        if normalize:
            hist = cv2.normalize(hist, hist).flatten()
        else:
            hist = hist.flatten()
        return hist
    
    # Color image - compute histogram for each channel
    hist_r = cv2.calcHist([image], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [bins], [0, 256])
    
    # Concatenate histograms
    hist = np.concatenate([hist_r, hist_g, hist_b]).flatten()
    
    if normalize:
        hist = hist / (image.shape[0] * image.shape[1])  # Normalize by image size
    
    return hist

def extract_color_moments(image):
    """
    Extract color moments (mean, std dev, skewness) for each channel.
    
    Parameters:
    - image: Input image (numpy array)
    
    Returns:
    - Array of color moments
    """
    # Convert to float for accurate calculations
    img_float = image.astype(np.float32)
    
    # Check if image is grayscale
    if len(image.shape) == 2 or image.shape[2] == 1:
        # Calculate moments for grayscale
        mean = np.mean(img_float)
        std = np.std(img_float)
        # Calculate skewness
        img_norm = img_float - mean
        skewness = np.mean(np.power(img_norm, 3)) / (np.power(std, 3) + 1e-6)  # Avoid division by zero
        return np.array([mean, std, skewness])
    
    # For color image, calculate moments for each channel
    moments = []
    for i in range(3):  # For each RGB channel
        channel = img_float[:,:,i]
        mean = np.mean(channel)
        std = np.std(channel)
        # Calculate skewness
        channel_norm = channel - mean
        skewness = np.mean(np.power(channel_norm, 3)) / (np.power(std, 3) + 1e-6)
        moments.extend([mean, std, skewness])
    
    return np.array(moments)

def extract_color_coherence_vector(image, bins=64):
    """
    Extract Color Coherence Vector (CCV) features.
    A CCV classifies pixels as coherent or incoherent based on whether
    they are part of a large similarly-colored region.
    
    Parameters:
    - image: Input image (numpy array)
    - bins: Number of bins per channel for quantization
    
    Returns:
    - CCV features
    """
    # Quantize colors
    quantized = np.floor(image / (256 / bins)).astype(np.int32)
    
    # Single index for each quantized color
    if len(image.shape) == 3:  # Color image
        color_index = (quantized[:,:,0] * bins * bins + 
                       quantized[:,:,1] * bins + 
                       quantized[:,:,2])
    else:  # Grayscale
        color_index = quantized
    
    # Initialize arrays for coherent and incoherent pixels
    coherent = np.zeros(bins**3 if len(image.shape) == 3 else bins)
    incoherent = np.zeros_like(coherent)
    
    # Define threshold for coherent regions
    threshold = 25  # Pixels in region
    
    # For each unique color
    for color in np.unique(color_index):
        # Create binary mask for this color
        mask = (color_index == color).astype(np.uint8)
        
        # Find connected components
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        
        # Count coherent and incoherent pixels
        for i in range(1, len(stats)):  # Skip background (0)
            size = stats[i, cv2.CC_STAT_AREA]
            if size >= threshold:
                coherent[color] += size
            else:
                incoherent[color] += size
    
    # Concatenate and return
    return np.concatenate([coherent, incoherent])

def extract_dominant_colors(image, n_colors=5):
    """
    Extract dominant colors using K-means clustering.
    
    Parameters:
    - image: Input image (numpy array)
    - n_colors: Number of dominant colors to extract
    
    Returns:
    - Array with dominant colors and their proportions
    """
    # Reshape image to a 2D array of pixels
    pixels = image.reshape(-1, 3) if len(image.shape) == 3 else image.reshape(-1, 1)
    
    # Convert to float for better precision
    pixels = pixels.astype(np.float32)
    
    # Define criteria and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Count occurrences of each label
    counts = np.bincount(labels.flatten())
    
    # Calculate proportions
    proportions = counts / len(pixels)
    
    # Combine colors and their proportions
    result = []
    for i in range(n_colors):
        color = centers[i].tolist()
        prop = proportions[i]
        result.extend(color + [prop])
    
    return np.array(result)