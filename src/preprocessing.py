"""Preprocessing module for Supermatcher v1.0 (Hybrid)

This module contains all image preprocessing functions for fingerprint processing:
- Image loading and normalization
- Segmentation (foreground/background separation)
- Coherence-enhancing diffusion for ridge enhancement

"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Tuple
import cv2
import numpy as np

from src.config import (
    NORMALISE_MEAN, NORMALISE_STD,
    SEGMENTATION_BLOCK_SIZE, SEGMENTATION_VARIANCE_THRESHOLD,
    COHERENCE_ITERATIONS, COHERENCE_KAPPA, COHERENCE_GAMMA
)


def load_grayscale_image(path: Path) -> np.ndarray:
    """Load fingerprint image as grayscale float32.
    
    Args:
        path: Path to fingerprint image file
        
    Returns:
        Grayscale image as float32 (0-255 range)
        
    Raises:
        FileNotFoundError: If image cannot be loaded
    """
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to read fingerprint image: {path}")
    
    return image.astype(np.float32)


def normalise_image(image: np.ndarray,
                    block_size: int = 16,
                    mean0: float = None,
                    var0: float = None) -> np.ndarray:
    """Normalize image intensity using local block statistics.
    
    Applies local normalization to compensate for uneven illumination and contrast.
    Each pixel is normalized based on the mean and variance of its local neighborhood.
    
    Args:
        image: Input grayscale image (any numeric type)
        block_size: Size of local neighborhood block (default 16)
        mean0: Target mean intensity (uses config default if None)
        var0: Target variance (uses config default if None)
        
    Returns:
        Normalized image (float32, 0-255 range)
        
    Note:
        Formula: normalized = mean0 + (image - local_mean) * sqrt(var0 / local_var)
    """
    if mean0 is None:
        mean0 = NORMALISE_MEAN
    if var0 is None:
        var0 = NORMALISE_STD ** 2  # Convert std to variance
    
    if image.dtype != np.float32:
        image = image.astype(np.float32)

    kernel = (block_size, block_size)
    local_mean = cv2.boxFilter(image, -1, kernel, normalize=True)
    local_sq_mean = cv2.boxFilter(image * image, -1, kernel, normalize=True)
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 1e-6)

    scale = np.sqrt(var0 / local_var)
    normalised = mean0 + (image - local_mean) * scale
    return np.clip(normalised, 0.0, 255.0)


def block_variance_segmentation(image: np.ndarray,
                                block_size: int = None,
                                threshold: float = None) -> np.ndarray:
    """Segment fingerprint foreground from background using block variance.
    
    Divides the image into blocks and computes variance for each block.
    High-variance blocks indicate ridge structures (foreground), while
    low-variance blocks indicate background or noise.
    
    Args:
        image: Input grayscale image (float32, 0-255)
        block_size: Size of square blocks (uses config default if None)
        threshold: Variance threshold (uses config default if None)
        
    Returns:
        Boolean mask: True for foreground, False for background
        
    Note:
        Applies morphological closing and opening to remove noise.
    """
    if block_size is None:
        block_size = SEGMENTATION_BLOCK_SIZE
    if threshold is None:
        threshold = SEGMENTATION_VARIANCE_THRESHOLD
    
    h, w = image.shape
    mask = np.zeros_like(image, dtype=np.uint8)

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = image[y:y + block_size, x:x + block_size]
            if block.size < block_size * block_size:
                continue
            variance = block.var()
            if variance >= threshold:
                mask[y:y + block_size, x:x + block_size] = 1

    # Morphological operations to clean up mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return mask.astype(bool)


def gaussian_derivatives(image: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute smoothed image gradients using Gaussian convolution.
    
    Applies Gaussian blur followed by Sobel operators to compute stable gradients.
    Uses float64 precision for numerical stability.
    
    Args:
        image: Input grayscale image (any numeric type)
        sigma: Standard deviation of Gaussian smoothing
        
    Returns:
        Tuple of (gx, gy) - gradient in x and y directions (float64)
    """
    ksize = int(6 * sigma + 1) | 1  # Ensure odd kernel size
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    return gx, gy


def coherence_diffusion(image: np.ndarray,
                       mask: np.ndarray,
                       iterations: int = None,
                       dt: float = 0.15,
                       grad_sigma: float = 1.0,
                       tensor_sigma: float = 2.0,
                       alpha: float = 0.01,
                       beta: float = 1.25) -> np.ndarray:
    """Apply coherence-enhancing diffusion (CED) for ridge enhancement.
    
    Implements anisotropic diffusion based on the local structure tensor, which
    enhances ridge structures while preserving edges. Diffusion is stronger along
    ridge directions and weaker across ridges.
    
    The algorithm:
    1. Computes structure tensor from image gradients
    2. Calculates eigenvalues (λ1, λ2) to determine local coherence
    3. Constructs diffusion tensor with anisotropic diffusivity
    4. Applies diffusion equation iteratively: I(t+1) = I(t) + dt * div(D·∇I)
    
    Args:
        image: Input grayscale image (float32, 0-255)
        mask: Boolean mask indicating valid foreground pixels
        iterations: Number of diffusion iterations (uses config default if None)
        dt: Time step size for numerical integration (default 0.15)
        grad_sigma: Sigma for Gaussian smoothing of gradients (default 1.0)
        tensor_sigma: Sigma for smoothing structure tensor (default 2.0)
        alpha: Minimum diffusivity perpendicular to ridges (default 0.01)
        beta: Maximum diffusivity parallel to ridges (default 1.25)
        
    Returns:
        Enhanced image (float32, 0-255) with connected ridges
        
    Note:
        Uses float64 precision internally for numerical stability.
        Coherence measure: exp(-(λ1-λ2)²/(λ1·λ2)) ∈ [0,1]
    """
    if iterations is None:
        iterations = COHERENCE_ITERATIONS
    
    I = image.astype(np.float64, copy=True)
    valid = mask.astype(bool)
    
    for _ in range(iterations):
        # Compute structure tensor
        gx, gy = gaussian_derivatives(I, grad_sigma)
        Jxx = cv2.GaussianBlur(gx * gx, (0, 0), tensor_sigma)
        Jyy = cv2.GaussianBlur(gy * gy, (0, 0), tensor_sigma)
        Jxy = cv2.GaussianBlur(gx * gy, (0, 0), tensor_sigma)

        # Eigenvalues of structure tensor
        trace = Jxx + Jyy
        diff = Jxx - Jyy
        discriminant = np.sqrt(np.maximum(diff * diff + 4.0 * Jxy * Jxy, 0.0))

        lambda1 = 0.5 * (trace + discriminant)
        lambda2 = 0.5 * (trace - discriminant)

        # Coherence measure
        denom = 1e-6 + lambda1 * lambda2 + 1e-6
        num = (lambda1 - lambda2) ** 2
        exponent = -np.divide(num, denom, out=np.zeros_like(num), where=denom != 0.0)
        exponent = np.clip(exponent, -60.0, 0.0)
        coherence = np.exp(exponent)
        
        # Diffusion coefficients
        c_parallel = alpha + (beta - alpha) * (1.0 - coherence)
        c_perp = alpha

        # Eigenvector (perpendicular to ridges)
        vx = 2.0 * Jxy
        vy = Jyy - Jxx + discriminant
        norm = np.sqrt(vx * vx + vy * vy)
        norm = np.where(norm < 1e-9, 1.0, norm)
        vx = vx / norm
        vy = vy / norm

        # Diffusion tensor
        Dxx = c_parallel * vx * vx + c_perp * vy * vy
        Dyy = c_parallel * vy * vy + c_perp * vx * vx
        Dxy = (c_parallel - c_perp) * vx * vy

        # Gradient of I
        Ix = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)

        # Flux J = D * ∇I
        Jx = Dxx * Ix + Dxy * Iy
        Jy = Dxy * Ix + Dyy * Iy

        # Divergence of flux
        divJx = cv2.Sobel(Jx, cv2.CV_64F, 1, 0, ksize=3)
        divJy = cv2.Sobel(Jy, cv2.CV_64F, 0, 1, ksize=3)
        divergence = divJx + divJy

        # Update only foreground pixels
        I[valid] = I[valid] + dt * divergence[valid]

    return np.clip(I, 0.0, 255.0).astype(np.float32)
