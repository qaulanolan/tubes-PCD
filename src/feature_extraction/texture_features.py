# src/feature_extraction/texture_features.py
"""
Module for extracting texture-based features from images.
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

def extract_glcm_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256):
    """
    Extract Gray Level Co-occurrence Matrix (GLCM) texture features.
    
    Parameters:
    - image: Input image (numpy array)
    - distances: List of distances between pixel pairs
    - angles: List of angles for pixel pairs
    - levels: Number of gray levels to quantize the image into
    
    Returns:
    - Array of GLCM features (contrast, dissimilarity, homogeneity, energy, correlation)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Normalize and quantize to reduce computation
    bins = 8  # Use 8 gray levels for efficiency
    gray = (gray / (256 / bins)).astype(np.uint8)
    
    # Compute GLCM
    glcm = graycomatrix(gray, distances=distances, angles=angles, 
                        levels=bins, symmetric=True, normed=True)
    
    # Extract features from GLCM
    features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        feature = graycoprops(glcm, prop).flatten()
        features.extend(feature)
    
    return np.array(features)

def extract_lbp_features(image, P=8, R=1, method='uniform'):
    """
    Extract Local Binary Pattern features for texture analysis.
    
    Parameters:
    - image: Input image (numpy array)
    - P: Number of circularly symmetric neighbor set points
    - R: Radius of the circle
    - method: LBP method {'default', 'ror', 'uniform', 'var'}
    
    Returns:
    - LBP histogram features
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Compute LBP
    lbp = local_binary_pattern(gray, P, R, method)
    
    # Compute histogram
    n_bins = P + 2 if method == 'uniform' else 2**P
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    return hist

def extract_haralick_features(image):
    """
    Extract Haralick texture features.
    
    Parameters:
    - image: Input image (numpy array)
    
    Returns:
    - Array of Haralick features
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Calculate texture features for 4 directions
    textures = []
    for angle in [0, 45, 90, 135]:
        # OpenCV doesn't have direct Haralick features, using mahotas
        try:
            import mahotas as mt
            features = mt.features.haralick(gray, distance=1, angle=angle, return_mean=True)
            textures.extend(features)
        except ImportError:
            # Fallback to basic statistics if mahotas is not available
            glcm = graycomatrix(gray, [1], [angle * np.pi/180], levels=256, symmetric=True, normed=True)
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
                feature = graycoprops(glcm, prop)[0, 0]
                textures.append(feature)
    
    return np.array(textures)

def extract_gabor_features(image, orientations=8, scales=5):
    """
    Extract Gabor filter-based texture features.
    
    Parameters:
    - image: Input image (numpy array)
    - orientations: Number of filter orientations
    - scales: Number of filter scales
    
    Returns:
    - Array of Gabor features
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    gray = gray.astype(np.float32)
    
    # Define Gabor filter parameters
    gabor_features = []
    
    for scale in range(scales):
        for orientation in range(orientations):
            # Create Gabor filter kernel
            wavelength = 2.0 ** scale
            theta = orientation * np.pi / orientations
            kernel = cv2.getGaborKernel(
                ksize=(31, 31),
                sigma=4.0,
                theta=theta,
                lambd=wavelength,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F
            )
            
            # Apply filter
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            
            # Calculate feature statistics
            mean = np.mean(filtered)
            std = np.std(filtered)
            gabor_features.extend([mean, std])
    
    return np.array(gabor_features)