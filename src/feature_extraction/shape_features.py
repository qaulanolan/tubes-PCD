# src/feature_extraction/shape_features.py
"""
Module for extracting shape-based features from images.
"""

import cv2
import numpy as np
from skimage.feature import hog

def extract_hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Extract Histogram of Oriented Gradients (HOG) features.
    
    Parameters:
    - image: Input image (numpy array)
    - orientations: Number of orientation bins
    - pixels_per_cell: Size (in pixels) of a cell
    - cells_per_block: Number of cells in each block
    
    Returns:
    - HOG features array
    """
    # Resize image for consistent feature size
    resized = cv2.resize(image, (128, 128))
    
    # Convert to grayscale if needed
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    else:
        gray = resized
    
    # Extract HOG features
    features = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm='L2-Hys',
        visualize=False
    )
    
    return features

def extract_contour_features(image):
    """
    Extract shape features based on contours.
    
    Parameters:
    - image: Input image (numpy array)
    
    Returns:
    - Shape features array
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Threshold the image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    features = []
    
    # If no contours found
    if not contours:
        # Return zeros
        return np.zeros(7)
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate area
    area = cv2.contourArea(largest_contour)
    
    # Calculate perimeter
    perimeter = cv2.arcLength(largest_contour, True)
    
    # Calculate circularity
    circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
    
    # Calculate bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Calculate aspect ratio
    aspect_ratio = float(w) / (h + 1e-6)
    
    # Calculate extent (area ratio)
    rect_area = w * h
    extent = float(area) / (rect_area + 1e-6)
    
    # Calculate moments and Hu moments
    moments = cv2.moments(largest_contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Log transform Hu moments (better for classification)
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-6)
    
    # Combine all features
    features = [area, perimeter, circularity, aspect_ratio, extent]
    features.extend(hu_moments)
    
    return np.array(features)

def extract_zernike_moments(image, radius=21, degree=8):
    """
    Extract Zernike moments for shape description.
    
    Parameters:
    - image: Input image (numpy array)
    - radius: Radius for Zernike polynomials
    - degree: Degree of Zernike polynomials
    
    Returns:
    - Zernike moments feature array
    """
    # Import mahotas here since it's a specialized library
    try:
        import mahotas as mt
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Threshold image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Calculate Zernike moments
        zernike = mt.features.zernike_moments(binary, radius, degree=degree)
        
        return zernike
    except ImportError:
        # Fallback to Hu moments if mahotas is not available
        return extract_contour_features(image)[-7:]  # Return just the Hu moments part

def extract_fourier_descriptors(image, n_descriptors=20):
    """
    Extract Fourier descriptors for shape representation.
    
    Parameters:
    - image: Input image (numpy array)
    - n_descriptors: Number of Fourier descriptors to return
    
    Returns:
    - Fourier descriptors array
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Threshold the image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours, return zeros
    if not contours:
        return np.zeros(n_descriptors * 2)
    
    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Convert contour to complex numbers
    contour_complex = np.empty(len(contour), dtype=complex)
    for i in range(len(contour)):
        contour_complex[i] = complex(contour[i][0][0], contour[i][0][1])
    
    # Compute Fourier descriptors
    fourier_result = np.fft.fft(contour_complex)
    
    # Get magnitude and normalize
    descriptors = np.abs(fourier_result)
    descriptors = descriptors / (descriptors[1] + 1e-6)  # Normalize by the first non-DC component
    
    # Take a subset of descriptors
    n_actual = min(n_descriptors, len(descriptors) // 2)
    descriptors = np.concatenate([descriptors[1:n_actual+1], descriptors[-n_actual:]])
    
    # Pad with zeros if needed
    if len(descriptors) < n_descriptors * 2:
        descriptors = np.pad(descriptors, (0, n_descriptors * 2 - len(descriptors)))
    
    return descriptors