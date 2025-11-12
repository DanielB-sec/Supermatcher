"""Feature extraction module for Supermatcher v1.0 (Hybrid)

This module contains all feature extraction functions for fingerprint processing:
- Ridge orientation and frequency estimation
- Log-Gabor filtering for ridge enhancement
- Binarization and skeletonization
- Minutiae extraction and validation
- Optional Level-3 features (pores)

"""

from __future__ import annotations
import math
from typing import List, Tuple, Sequence
import cv2
import numpy as np

from src.models import Minutia, Pore
from src.config import (
    ORIENTATION_BLOCK_SIZE,
    LOG_GABOR_KERNEL_SIZE, LOG_GABOR_SIGMA_ONEF,
    BINARISATION_METHOD, THINNING_MAX_ITER,
    MINUTIAE_BORDER_MARGIN, MINUTIA_PAIR_DISTANCE,
    PORE_MIN_RADIUS, PORE_MAX_RADIUS, PORE_MIN_STRENGTH,
    FREQUENCY_MIN, FREQUENCY_MAX
)


def estimate_orientation_and_frequency(image: np.ndarray,
                                       mask: np.ndarray,
                                       block_size: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate ridge orientation and spatial frequency for each block.
    
    Args:
        image: Enhanced fingerprint image (float32, 0-255)
        mask: Boolean foreground mask
        block_size: Analysis block size (uses config default if None)
        
    Returns:
        Tuple of (orientation, frequency):
        - orientation: Ridge angle in radians, shape (blocks_y, blocks_x)
        - frequency: Spatial frequency in cycles/pixel, shape (blocks_y, blocks_x)
    """
    if block_size is None:
        block_size = ORIENTATION_BLOCK_SIZE
    
    h, w = image.shape
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    blocks_y = h // block_size
    blocks_x = w // block_size
    orientation = np.zeros((blocks_y, blocks_x), dtype=np.float32)
    frequency = np.zeros_like(orientation)

    mask_uint8 = mask.astype(np.uint8)

    for by in range(blocks_y):
        for bx in range(blocks_x):
            y0 = by * block_size
            x0 = bx * block_size
            block_mask = mask_uint8[y0:y0 + block_size, x0:x0 + block_size]
            
            if block_mask.sum() < 0.6 * (block_size * block_size):
                continue

            gx_block = gx[y0:y0 + block_size, x0:x0 + block_size]
            gy_block = gy[y0:y0 + block_size, x0:x0 + block_size]

            vxy = 2.0 * np.sum(gx_block * gy_block)
            vxx_yy = np.sum(gx_block * gx_block - gy_block * gy_block)
            theta = 0.5 * math.atan2(vxy, vxx_yy)
            
            if not np.isfinite(theta):
                theta = 0.0
            orientation[by, bx] = theta

            block = image[y0:y0 + block_size, x0:x0 + block_size]
            freq = estimate_ridge_frequency(block, theta)
            
            if not np.isfinite(freq) or freq <= 0.0:
                freq = 0.0
            frequency[by, bx] = freq

    orientation = np.nan_to_num(orientation, nan=0.0, posinf=0.0, neginf=0.0)
    frequency = np.nan_to_num(frequency, nan=0.0, posinf=0.0, neginf=0.0)
    
    return orientation, frequency


def estimate_ridge_frequency(block: np.ndarray, theta: float) -> float:
    """Estimate local ridge frequency from 1D projection orthogonal to orientation.
    
    Args:
        block: Image block (float32, typically 16×16 pixels)
        theta: Ridge orientation in radians
        
    Returns:
        Ridge frequency in cycles per pixel (0.0 if invalid)
    """
    if not np.isfinite(theta):
        return 0.0
    
    h, w = block.shape
    center = np.array([(w - 1) / 2.0, (h - 1) / 2.0], dtype=np.float32)

    normal = np.array([math.sin(theta), -math.cos(theta)], dtype=np.float32)

    y_indices, x_indices = np.mgrid[0:h, 0:w]
    coords = np.stack([x_indices, y_indices], axis=-1).reshape(-1, 2).astype(np.float32)
    rel = coords - center
    projection = rel @ normal
    bins = np.round(projection).astype(int)

    min_bin = bins.min()
    max_bin = bins.max()
    length = max_bin - min_bin + 1
    
    if length < 8:
        return 0.0

    profile = np.zeros(length, dtype=np.float32)
    counts = np.zeros(length, dtype=np.float32)
    
    for value, bin_id in zip(block.flatten(), bins):
        index = bin_id - min_bin
        profile[index] += value
        counts[index] += 1.0

    counts[counts == 0.0] = 1.0
    profile /= counts

    profile -= profile.mean()
    
    if np.allclose(profile.var(), 0.0):
        return 0.0

    fft = np.fft.rfft(profile)
    magnitudes = np.abs(fft)
    magnitudes[0] = 0.0

    peak_index = np.argmax(magnitudes)
    
    if peak_index == 0:
        return 0.0

    frequency = peak_index / float(length)
    return float(np.clip(frequency, FREQUENCY_MIN, FREQUENCY_MAX))


def create_log_gabor_kernel(size: int,
                            f0: float,
                            theta0: float,
                            sigma_r: float = 1.5,
                            sigma_theta_deg: float = 12.0) -> np.ndarray:
    """Construct a Log-Gabor filter kernel in frequency domain.
    
    Args:
        size: Kernel size (should match block_size)
        f0: Center frequency in cycles/pixel
        theta0: Ridge orientation in radians
        sigma_r: Radial bandwidth parameter (default 1.5)
        sigma_theta_deg: Angular bandwidth in degrees (default 12)
        
    Returns:
        Complex-valued frequency-domain filter kernel
    """
    if f0 <= 0.0:
        return np.zeros((size, size), dtype=np.complex64)

    fy = np.fft.fftfreq(size)
    fx = np.fft.fftfreq(size)
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX ** 2 + FY ** 2)
    Theta = np.arctan2(FY, FX)

    sigma_theta = math.radians(sigma_theta_deg)
    theta_diff = np.mod(Theta - theta0 + math.pi, 2.0 * math.pi) - math.pi

    radial = np.zeros_like(R)
    valid = R > 0
    radial[valid] = np.exp(-((np.log(R[valid] / f0)) ** 2) / (2 * (np.log(sigma_r) ** 2)))

    angular = np.exp(-(theta_diff ** 2) / (2 * sigma_theta ** 2))
    filter_kernel = radial * angular
    
    return filter_kernel.astype(np.complex64)


def apply_log_gabor_enhancement(image: np.ndarray,
                                mask: np.ndarray,
                                orientation: np.ndarray,
                                frequency: np.ndarray,
                                block_size: int = None,
                                scales: Tuple[float, ...] = (0.85, 1.0, 1.2)) -> np.ndarray:
    """Enhance fingerprint ridges using Log-Gabor filters.
    
    Args:
        image: Diffusion-enhanced image (float32, 0-255)
        mask: Boolean foreground mask
        orientation: Ridge orientation map (radians)
        frequency: Ridge frequency map (cycles/pixel)
        block_size: Processing block size (uses config default if None)
        scales: Frequency scale factors (default [0.85, 1.0, 1.2])
        
    Returns:
        Enhanced fingerprint image (float32, 0-255)
    """
    if block_size is None:
        block_size = ORIENTATION_BLOCK_SIZE
    
    h, w = image.shape
    enhanced = np.zeros_like(image)
    weights = np.zeros_like(image)
    
    window = np.outer(np.hanning(block_size), np.hanning(block_size)).astype(np.float32)
    window /= window.max() + 1e-6

    for by in range(orientation.shape[0]):
        for bx in range(orientation.shape[1]):
            y0 = by * block_size
            x0 = bx * block_size
            block = image[y0:y0 + block_size, x0:x0 + block_size]
            block_mask = mask[y0:y0 + block_size, x0:x0 + block_size]
            
            if block.shape != (block_size, block_size):
                continue
            
            if block_mask.sum() < 0.5 * block_size * block_size:
                continue

            theta0 = orientation[by, bx]
            f0 = frequency[by, bx]
            
            if f0 <= 0:
                f0 = 0.1

            block_fft = np.fft.fft2(block * window)
            accum = np.zeros_like(block, dtype=np.float32)
            
            for scale in scales:
                kernel = create_log_gabor_kernel(block_size, f0 * scale, theta0)
                response = np.fft.ifft2(block_fft * kernel)
                accum += np.real(response)

            enhanced_block = accum / float(len(scales))
            enhanced[y0:y0 + block_size, x0:x0 + block_size] += enhanced_block * window
            weights[y0:y0 + block_size, x0:x0 + block_size] += window

    weights[weights == 0.0] = 1.0
    merged = enhanced / weights
    merged[~mask] = image[~mask]
    
    return np.clip(merged, 0.0, 255.0)


def binarise_and_thin(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Binarize enhanced image and compute skeleton.
    
    Args:
        image: Enhanced fingerprint image (float32, 0-255)
        mask: Boolean foreground mask
        
    Returns:
        Tuple of (binary, skeleton):
        - binary: Binary image with ridges=1, valleys=0 (uint8)
        - skeleton: Thinned 1-pixel wide skeleton (uint8)
    """
    clipped = image.copy()
    clipped[~mask] = 0.0
    clipped_uint8 = np.clip(clipped, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(clipped_uint8, 0, 255, BINARISATION_METHOD)
    binary = (binary > 0).astype(np.uint8)

    skeleton = zhang_suen_thinning(binary)
    skeleton &= mask.astype(np.uint8)
    
    return binary, skeleton


def zhang_suen_thinning(binary: np.ndarray, max_iter: int = None) -> np.ndarray:
    """Apply Zhang-Suen thinning algorithm to obtain 1-pixel wide skeleton.
    
    Args:
        binary: Binary ridge image (uint8, 0 or 1)
        max_iter: Maximum iterations (uses config default if None)
        
    Returns:
        Thinned skeleton (uint8, 0 or 1)
    """
    if max_iter is None:
        max_iter = THINNING_MAX_ITER
    
    img = binary.copy().astype(np.uint8)
    changed = True
    iteration = 0
    
    while changed and iteration < max_iter:
        changed = False
        iteration += 1
        
        for step in (0, 1):
            markers = np.zeros_like(img)
            
            for y in range(1, img.shape[0] - 1):
                for x in range(1, img.shape[1] - 1):
                    if img[y, x] != 1:
                        continue
                    
                    neighbourhood = img[y - 1:y + 2, x - 1:x + 2]
                    neighbours = [
                        neighbourhood[0, 1], neighbourhood[0, 2], neighbourhood[1, 2],
                        neighbourhood[2, 2], neighbourhood[2, 1], neighbourhood[2, 0],
                        neighbourhood[1, 0], neighbourhood[0, 0],
                    ]
                    
                    transitions = sum((neighbours[i] == 0 and neighbours[(i + 1) % 8] == 1) for i in range(8))
                    count = sum(neighbours)
                    
                    if transitions != 1 or not (2 <= count <= 6):
                        continue

                    if step == 0:
                        if neighbours[0] * neighbours[2] * neighbours[4] != 0:
                            continue
                        if neighbours[2] * neighbours[4] * neighbours[6] != 0:
                            continue
                    else:
                        if neighbours[0] * neighbours[2] * neighbours[6] != 0:
                            continue
                        if neighbours[0] * neighbours[4] * neighbours[6] != 0:
                            continue
                    
                    markers[y, x] = 1
            
            img[markers == 1] = 0
            if markers.any():
                changed = True
    
    return img


def extract_minutiae(skeleton: np.ndarray,
                     mask: np.ndarray,
                     orientation: np.ndarray,
                     block_size: int = None) -> List[Minutia]:
    """Extract minutiae points using Crossing Number (CN) method.
    
    Args:
        skeleton: Thinned binary skeleton (uint8, 0 or 1)
        mask: Boolean foreground mask
        orientation: Ridge orientation map (radians)
        block_size: Block size for orientation lookup (uses config default if None)
        
    Returns:
        List of Minutia objects with (x, y, angle, type, quality)
    """
    if block_size is None:
        block_size = ORIENTATION_BLOCK_SIZE
    
    minutiae: List[Minutia] = []
    h, w = skeleton.shape

    def block_orientation(y: int, x: int) -> float:
        by = min(max(y // block_size, 0), orientation.shape[0] - 1)
        bx = min(max(x // block_size, 0), orientation.shape[1] - 1)
        return orientation[by, bx]

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skeleton[y, x] == 0 or not mask[y, x]:
                continue
            
            neighbourhood = skeleton[y - 1:y + 2, x - 1:x + 2]
            
            if neighbourhood.sum() < 2:
                continue
            
            p2, p3, p4 = neighbourhood[0, 1], neighbourhood[0, 2], neighbourhood[1, 2]
            p5, p6, p7 = neighbourhood[2, 2], neighbourhood[2, 1], neighbourhood[2, 0]
            p8, p9 = neighbourhood[1, 0], neighbourhood[0, 0]
            neighbours = [p2, p3, p4, p5, p6, p7, p8, p9]
            
            transitions = sum((neighbours[i] == 0 and neighbours[(i + 1) % 8] == 1) for i in range(8))

            if transitions == 1:
                kind = "ending"
            elif transitions == 3:
                kind = "bifurcation"
            else:
                continue

            if min(x, w - 1 - x, y, h - 1 - y) < MINUTIAE_BORDER_MARGIN:
                continue

            theta = block_orientation(y, x)
            neighbourhood_var = neighbourhood.var()
            minutiae.append(Minutia(float(x), float(y), theta, kind, float(neighbourhood_var)))

    return minutiae


def validate_minutiae(minutiae: List[Minutia],
                      skeleton: np.ndarray,
                      mask: np.ndarray,
                      validate_with_context: bool = True) -> List[Minutia]:
    """Remove spurious minutiae using spatial and structural heuristics.
    
    Filters out false minutiae caused by noise, broken ridges, or artifacts.
    Applies multiple validation criteria:
    1. Mask-based: Remove minutiae in background regions
    2. Distance-based: Remove minutiae too close to each other (likely noise)
    3. Angular consistency: Check if minutia angle aligns with local ridge orientation
    4. Neighborhood quality: Verify surrounding skeleton structure is consistent
    
    Args:
        minutiae: List of extracted minutiae (from extract_minutiae)
        skeleton: Thinned ridge skeleton used for context validation
        mask: Boolean foreground mask
        validate_with_context: If True, apply neighborhood and angular consistency checks
    
    Returns:
        Filtered list of valid minutiae with improved quality estimates
    
    Note:
        Removes minutiae pairs closer than MINUTIA_PAIR_DISTANCE (default 12 pixels).
        Angular consistency checks use 45 degrees threshold.
    """
    if not minutiae:
        return []

    # First pass: mask-based filtering
    filtered: List[Minutia] = []
    for m in minutiae:
        if not mask[int(m.y), int(m.x)]:
            continue
        filtered.append(m)
    
    # Second pass: context-based validation (neighborhood + angular consistency)
    if validate_with_context:
        h, w = skeleton.shape
        validated: List[Minutia] = []
        
        for m in filtered:
            quality = m.quality
            x_int, y_int = int(m.x), int(m.y)
            
            # Sample skeleton in 12-pixel radius
            radius = 12
            y_min = max(0, y_int - radius)
            y_max = min(h, y_int + radius + 1)
            x_min = max(0, x_int - radius)
            x_max = min(w, x_int + radius + 1)
            
            # Count active skeleton pixels
            neighborhood = skeleton[y_min:y_max, x_min:x_max]
            skeleton_pixel_count = int(np.sum(neighborhood > 0))
            
            if skeleton_pixel_count < 8:
                # Insufficient ridge support, penalize quality
                quality *= 0.7
            
            # Check angular consistency in 15-pixel radius
            radius_ori = 15
            y_min_ori = max(0, y_int - radius_ori)
            y_max_ori = min(h, y_int + radius_ori + 1)
            x_min_ori = max(0, x_int - radius_ori)
            x_max_ori = min(w, x_int + radius_ori + 1)
            
            # Compute mean orientation from nearby skeleton pixels
            ridge_patch = skeleton[y_min_ori:y_max_ori, x_min_ori:x_max_ori]
            ridge_coords = np.argwhere(ridge_patch > 0)
            
            if len(ridge_coords) > 3:
                # Estimate local ridge direction using gradient
                gy, gx = np.gradient(ridge_patch.astype(np.float32))
                mean_angle = float(np.arctan2(gy.mean(), gx.mean()))
                
                # Compute angle difference
                angle_diff = abs(mean_angle - m.angle)
                angle_diff = min(angle_diff, 2.0 * math.pi - angle_diff)
                
                if angle_diff > math.radians(45.0):
                    # Large angle inconsistency, penalize quality
                    quality *= 0.8
            
            # Create validated minutia with updated quality
            validated.append(Minutia(
                x=m.x,
                y=m.y,
                angle=m.angle,
                kind=m.kind,
                quality=quality,
            ))
        
        filtered = validated

    # Third pass: spatial proximity filtering
    filtered.sort(key=lambda m: m.quality, reverse=True)
    accepted: List[Minutia] = []
    for m in filtered:
        too_close = False
        for n in accepted:
            dx = m.x - n.x
            dy = m.y - n.y
            distance = math.hypot(dx, dy)
            if distance < MINUTIA_PAIR_DISTANCE and m.kind == n.kind:
                too_close = True
                break
        if not too_close:
            accepted.append(m)

    return accepted


def detect_pores(image: np.ndarray,
                 mask: np.ndarray,
                 min_radius: float = None,
                 max_radius: float = None,
                 threshold: float = None) -> List[Pore]:
    """Detect sweat pores using multi-scale Laplacian-of-Gaussian (LoG) filter.
    
    Args:
        image: Enhanced fingerprint image (float32, 0-255)
        mask: Boolean foreground mask
        min_radius: Minimum pore radius (uses config default if None)
        max_radius: Maximum pore radius (uses config default if None)
        threshold: Minimum detection strength (uses config default if None)
        
    Returns:
        List of Pore objects with (x, y, radius, strength)
    """
    if min_radius is None:
        min_radius = PORE_MIN_RADIUS
    if max_radius is None:
        max_radius = PORE_MAX_RADIUS
    if threshold is None:
        threshold = PORE_MIN_STRENGTH
    
    pores: List[Pore] = []
    sigmas = np.linspace(min_radius, max_radius, 5)
    
    for sigma in sigmas:
        ksize = int(6 * sigma + 1) | 1
        blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)
        lap = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
        response = -lap * (sigma ** 2)
        
        local_max = (response == cv2.dilate(response, np.ones((3, 3))))
        candidates = np.where((response > threshold) & local_max & mask)
        
        for y, x in zip(*candidates):
            pores.append(Pore(float(x), float(y), float(sigma), float(response[y, x])))
    
    return pores
