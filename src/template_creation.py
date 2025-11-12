"""Template creation and fusion module for Supermatcher v1.0 (Hybrid)

This module contains:
- CancelableHasher: Protected template generation using random projection
- Feature vector fusion (quality-weighted averaging)
- Minutiae fusion (spatial clustering with RANSAC alignment)
- Template quality assessment
- Template I/O (save/load)

"""

from __future__ import annotations
import hashlib
import math
import pickle
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import cv2
import numpy as np
from scipy.spatial import KDTree

from src.models import Minutia, Pore, FingerprintTemplate, FusionSettings
from src.config import (
    FEATURE_VECTOR_DIM, MAX_MINUTIAE_ENCODER, MINUTIA_FEATURE_SIZE,
    ORI_HIST_BINS, FREQ_HIST_BINS, GABOR_KERNELS,
    FUSION_RANSAC_ITERATIONS, FUSION_RANSAC_THRESHOLD,
    FUSION_ALIGNMENT_THRESHOLD, FUSION_MINUTIAE_CLUSTER_RADIUS,
    FUSION_ANGLE_TOLERANCE, DEFAULT_FUSION_MIN_CONSENSUS, FUSION_MIN_QUALITY,
    TEMPLATE_EXTENSION
)


# ---------------------------------------------------------------------------
# Cancelable Hasher


def _derive_seed_from_key(key: str, feature_dim: int, projection_dim: int) -> np.random.Generator:
    """Derive deterministic random seed from user key and dimensions."""
    key_material = f"{key}|{feature_dim}|{projection_dim}".encode("utf-8")
    digest = hashlib.sha256(key_material).digest()
    seed = int.from_bytes(digest[:8], "big", signed=False)
    return np.random.default_rng(seed)


class CancelableHasher:
    """Cancelable biometric template encoder using random projection and binarization.
    
    Implements: hash = sign(Px + b), where P is key-derived random matrix.
    
    Properties:
    - Non-invertible: Cannot recover features from hash
    - Revocable: Different key → different template
    - Renewable: Multiple independent templates per user
    - Similarity-preserving: Hamming distance correlates with feature distance
    
    Attributes:
        feature_dim: Input vector dimension (default 736)
        projection_dim: Output hash dimension before packing (default 512 bits)
        hash_count: Number of independent hashes (default 2)
        bit_length: Total bits (projection_dim × hash_count)
    """
    
    def __init__(self, feature_dim: int, projection_dim: int, key: str, hash_count: int = 1) -> None:
        if projection_dim <= 0 or hash_count <= 0:
            raise ValueError("projection_dim and hash_count must be positive")
        
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
        self.hash_count = hash_count
        self._projections: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []
        
        for index in range(hash_count):
            seed_key = f"{key}:{index}"
            rng = _derive_seed_from_key(seed_key, feature_dim, projection_dim)
            
            projection = rng.normal(0.0, 1.0 / math.sqrt(projection_dim),
                                   size=(projection_dim, feature_dim)).astype(np.float32)
            bias = rng.normal(0.0, 0.05, size=(projection_dim,)).astype(np.float32)
            
            self._projections.append(projection)
            self._biases.append(bias)
        
        self._pack_length = (self.bit_length + 7) // 8

    @property
    def bit_length(self) -> int:
        return self.projection_dim * self.hash_count

    def encode(self, features: np.ndarray) -> np.ndarray:
        """Encode feature vector into packed binary template."""
        vector = np.asarray(features, dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] != self.feature_dim:
            raise ValueError(f"Expected 1D vector of length {self.feature_dim}")
        
        bits_accum = []
        for projection, bias in zip(self._projections, self._biases):
            projected = (projection @ vector) + bias
            bits_accum.append((projected >= 0.0).astype(np.uint8))
        
        concatenated = np.concatenate(bits_accum, axis=0)
        return np.packbits(concatenated)

    def similarity(self, packed_a: np.ndarray, packed_b: np.ndarray) -> float:
        """Compute similarity between two packed templates (1 - hamming_dist/total_bits)."""
        a = np.asarray(packed_a, dtype=np.uint8)
        b = np.asarray(packed_b, dtype=np.uint8)
        
        if a.shape != b.shape or a.size != self._pack_length:
            raise ValueError("Packed template size mismatch")
        
        xor = np.bitwise_xor(a, b)
        mismatches = np.unpackbits(xor)[:self.bit_length]
        reshaped = mismatches.reshape(self.hash_count, self.projection_dim)
        per_hash_similarity = 1.0 - (reshaped.sum(axis=1) / float(self.projection_dim))
        return float(per_hash_similarity.mean())


# ---------------------------------------------------------------------------
# Helper Functions


def _angle_difference(theta1: float, theta2: float) -> float:
    """Compute absolute angular difference in [0, π] radians."""
    diff = (theta1 - theta2 + math.pi) % (2.0 * math.pi) - math.pi
    return abs(diff)


# ---------------------------------------------------------------------------
# Feature Vector Fusion


def fuse_feature_vectors(templates: Sequence[FingerprintTemplate]) -> Optional[np.ndarray]:
    """Fuse feature vectors using quality-weighted averaging.
    
    Args:
        templates: Sequence of FingerprintTemplate objects with raw_features
        
    Returns:
        Fused feature vector (L2-normalized), or None if no valid vectors
    """
    vectors: List[np.ndarray] = []
    weights: List[float] = []
    
    for template in templates:
        if template.raw_features is None:
            continue
        vectors.append(np.asarray(template.raw_features, dtype=np.float32))
        weights.append(max(template.quality, 1e-6))
    
    if not vectors:
        return None
    
    stacked = np.stack(vectors, axis=0)
    weights_arr = np.asarray(weights, dtype=np.float32)
    
    if np.allclose(weights_arr.sum(), 0.0):
        weights_arr = np.ones_like(weights_arr) / float(len(weights_arr))
    else:
        weights_arr /= weights_arr.sum()
    
    fused = weights_arr @ stacked
    norm = float(np.linalg.norm(fused))
    if norm > 0.0:
        fused = fused / norm
    
    return fused.astype(np.float32)


# ---------------------------------------------------------------------------
# RANSAC Alignment


def estimate_rigid_transform_ransac(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    max_iterations: int = None,
    threshold: float = None,
) -> Tuple[Optional[np.ndarray], float]:
    """Estimate rigid transformation using RANSAC.
    
    Args:
        src_points: Source points (N, 2)
        dst_points: Destination points (N, 2)
        max_iterations: Max RANSAC iterations (uses config default if None)
        threshold: Inlier distance threshold (uses config default if None)
        
    Returns:
        Tuple of (transformation_matrix 2×3, confidence_score)
    """
    if max_iterations is None:
        max_iterations = FUSION_RANSAC_ITERATIONS
    if threshold is None:
        threshold = FUSION_RANSAC_THRESHOLD
    
    if src_points.shape[0] < 3 or src_points.shape != dst_points.shape:
        return None, 0.0
    
    best_transform = None
    best_inliers = 0
    n_points = src_points.shape[0]
    
    for _ in range(max_iterations):
        if n_points < 2:
            break
        indices = np.random.choice(n_points, size=min(2, n_points), replace=False)
        src_sample = src_points[indices]
        dst_sample = dst_points[indices]
        
        transform = cv2.estimateAffinePartial2D(
            src_sample.reshape(-1, 1, 2).astype(np.float32),
            dst_sample.reshape(-1, 1, 2).astype(np.float32),
        )
        
        if transform is None or transform[0] is None:
            continue
        
        T = transform[0]
        src_transformed = cv2.transform(src_points.reshape(-1, 1, 2).astype(np.float32), T)
        src_transformed = src_transformed.reshape(-1, 2)
        
        distances = np.linalg.norm(src_transformed - dst_points, axis=1)
        inliers = np.sum(distances < threshold)
        
        if inliers > best_inliers:
            best_inliers = inliers
            best_transform = T
    
    if best_transform is None:
        return None, 0.0
    
    confidence = float(best_inliers) / float(n_points)
    return best_transform, confidence


def align_minutiae_sets(
    reference_minutiae: Sequence[Minutia],
    target_minutiae: Sequence[Minutia],
    image_shape: Tuple[int, int],
) -> Tuple[List[Minutia], float]:
    """Align target minutiae to reference using RANSAC.
    
    Args:
        reference_minutiae: Reference minutiae list
        target_minutiae: Target minutiae to transform
        image_shape: Image dimensions (height, width)
        
    Returns:
        Tuple of (aligned_minutiae, alignment_confidence)
    """
    if len(reference_minutiae) < 3 or len(target_minutiae) < 3:
        return list(target_minutiae), 0.0
    
    ref_points = np.array([[m.x, m.y] for m in reference_minutiae], dtype=np.float32)
    tgt_points = np.array([[m.x, m.y] for m in target_minutiae], dtype=np.float32)
    
    tree = KDTree(ref_points)
    distances, indices = tree.query(tgt_points, k=1)
    
    valid = distances < FUSION_ALIGNMENT_THRESHOLD
    if valid.sum() < 3:
        return list(target_minutiae), 0.0
    
    src_matched = tgt_points[valid]
    dst_matched = ref_points[indices[valid]]
    
    transform, confidence = estimate_rigid_transform_ransac(src_matched, dst_matched)
    
    if transform is None:
        return list(target_minutiae), 0.0
    
    aligned_minutiae: List[Minutia] = []
    for minutia in target_minutiae:
        point = np.array([[minutia.x, minutia.y]], dtype=np.float32).reshape(1, 1, 2)
        transformed = cv2.transform(point, transform).reshape(2)
        
        rotation = math.atan2(transform[1, 0], transform[0, 0])
        new_angle = (minutia.angle + rotation) % (2.0 * math.pi)
        
        aligned_minutiae.append(Minutia(
            x=float(transformed[0]),
            y=float(transformed[1]),
            angle=new_angle,
            kind=minutia.kind,
            quality=minutia.quality,
        ))
    
    return aligned_minutiae, confidence


# ---------------------------------------------------------------------------
# Minutiae Fusion


def fuse_minutiae_consensus(
    templates: Sequence[FingerprintTemplate],
    settings: FusionSettings,
) -> List[Minutia]:
    """Fuse minutiae from multiple templates using spatial clustering.
    
    Args:
        templates: Sequence of FingerprintTemplate objects with minutiae
        settings: Fusion configuration (distance, angle_rad, min_consensus)
        
    Returns:
        List of fused minutiae with consensus quality scores
    """
    if not templates:
        return []
    
    clusters: List[dict] = []
    
    for sample_idx, template in enumerate(templates):
        sample_quality = max(float(template.quality), 1e-6)
        for minutia in template.minutiae or []:
            x, y, angle = float(minutia.x), float(minutia.y), float(minutia.angle)
            local_quality = float(minutia.quality)
            
            if not math.isfinite(local_quality) or local_quality <= 0.0:
                local_quality = 1.0
            else:
                local_quality = min(local_quality, 255.0)
            
            weight = sample_quality * (local_quality / 255.0)
            if weight <= 0.0:
                continue
            
            assigned = False
            for cluster in clusters:
                dist = math.hypot(x - cluster["centroid_x"], y - cluster["centroid_y"])
                if dist > settings.distance:
                    continue
                
                angle_diff = _angle_difference(angle, cluster["mean_angle"])
                if angle_diff > math.radians(settings.angle_deg):
                    continue
                
                cluster["sum_weights"] += weight
                cluster["sum_x"] += weight * x
                cluster["sum_y"] += weight * y
                cluster["sum_cos"] += weight * math.cos(angle)
                cluster["sum_sin"] += weight * math.sin(angle)
                cluster["sample_ids"].add(sample_idx)
                cluster["kind_weights"][minutia.kind] = cluster["kind_weights"].get(minutia.kind, 0.0) + weight
                
                if cluster["sum_weights"] > 0.0:
                    cluster["centroid_x"] = cluster["sum_x"] / cluster["sum_weights"]
                    cluster["centroid_y"] = cluster["sum_y"] / cluster["sum_weights"]
                    cluster["mean_angle"] = math.atan2(cluster["sum_sin"], cluster["sum_cos"])
                
                assigned = True
                break
            
            if not assigned:
                clusters.append({
                    "sum_weights": weight,
                    "sum_x": weight * x,
                    "sum_y": weight * y,
                    "sum_cos": weight * math.cos(angle),
                    "sum_sin": weight * math.sin(angle),
                    "centroid_x": x,
                    "centroid_y": y,
                    "mean_angle": angle,
                    "sample_ids": {sample_idx},
                    "kind_weights": {minutia.kind: weight},
                })
    
    total_samples = len(templates)
    fused_minutiae: List[Minutia] = []
    
    for cluster in clusters:
        if cluster["sum_weights"] <= 0.0:
            continue
        
        consensus_ratio = len(cluster["sample_ids"]) / float(total_samples)
        if consensus_ratio < settings.min_consensus:
            continue
        
        x = cluster["sum_x"] / cluster["sum_weights"]
        y = cluster["sum_y"] / cluster["sum_weights"]
        angle = math.atan2(cluster["sum_sin"], cluster["sum_cos"])
        
        kind_weights = cluster["kind_weights"]
        kind = max(kind_weights.items(), key=lambda item: item[1])[0] if kind_weights else "ending"
        
        fused_minutiae.append(Minutia(
            x=float(x),
            y=float(y),
            angle=float(angle),
            kind=kind,
            quality=float(np.clip(consensus_ratio, 0.0, 1.0)),
        ))
    
    return fused_minutiae


def fuse_protected_templates(
    templates: Sequence[FingerprintTemplate],
    hasher: CancelableHasher,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Fuse protected templates using majority voting on bits.
    
    Args:
        templates: Sequence of FingerprintTemplate with protected hashes
        hasher: CancelableHasher instance for bit length
        
    Returns:
        Tuple of (fused_bits, tie_mask) or (None, None)
    """
    if not templates:
        return None, None
    
    bit_length = hasher.bit_length
    unpacked: List[np.ndarray] = []
    weights: List[float] = []
    
    for template in templates:
        packed = np.asarray(template.protected, dtype=np.uint8)
        if packed.size == 0:
            continue
        bits = np.unpackbits(packed, bitorder="big")[:bit_length]
        unpacked.append(bits.astype(np.float32, copy=False))
        weights.append(max(template.quality, 1e-6))
    
    if not unpacked:
        return None, None
    
    weights_arr = np.asarray(weights, dtype=np.float32)
    if np.allclose(weights_arr.sum(), 0.0):
        weights_arr = np.ones_like(weights_arr) / float(len(weights_arr))
    else:
        weights_arr /= weights_arr.sum()
    
    stacked = np.stack(unpacked, axis=0)
    weighted = weights_arr @ stacked
    fused_bits = (weighted > 0.5).astype(np.uint8)
    tie_mask = np.isclose(weighted, 0.5, atol=1e-6)
    
    return fused_bits, tie_mask


# ---------------------------------------------------------------------------
# Template Creation


def create_fused_template(
    identifier: str,
    templates: Sequence[FingerprintTemplate],
    hasher: CancelableHasher,
    settings: FusionSettings,
) -> Optional[FingerprintTemplate]:
    """Create fused template from multiple samples of same identity.
    
    Args:
        identifier: User identifier
        templates: Sequence of templates to fuse
        hasher: CancelableHasher for protected template encoding
        settings: Fusion configuration
        
    Returns:
        Fused FingerprintTemplate or None if fusion fails
    """
    if not templates:
        return None
    
    fused_vector = fuse_feature_vectors(templates)
    if fused_vector is None:
        return None
    
    fused_minutiae = fuse_minutiae_consensus(templates, settings)
    
    qualities = np.array([max(t.quality, 1e-6) for t in templates], dtype=np.float32)
    if np.allclose(qualities.sum(), 0.0):
        template_weights = np.ones_like(qualities) / float(len(qualities))
    else:
        template_weights = qualities / qualities.sum()
    
    avg_quality = float(np.clip(np.dot(template_weights, np.array([t.quality for t in templates])), 0.0, 1.0))
    consensus_avg = float(np.mean([m.quality for m in fused_minutiae])) if fused_minutiae else 0.0
    
    encode_from_vector = hasher.encode(fused_vector)
    fused_bits, tie_mask = fuse_protected_templates(templates, hasher)
    
    if fused_bits is None:
        protected = encode_from_vector
    else:
        bit_length = hasher.bit_length
        pad = (-bit_length) % 8
        majority_bits = fused_bits
        fallback_bits = np.unpackbits(encode_from_vector, bitorder="big")[:bit_length]
        
        if tie_mask is not None and tie_mask.any():
            majority_bits = majority_bits.copy()
            majority_bits[tie_mask] = fallback_bits[tie_mask]
        
        if pad:
            packed_bits = np.concatenate([majority_bits, np.zeros(pad, dtype=np.uint8)])
        else:
            packed_bits = majority_bits
        
        protected = np.packbits(packed_bits, bitorder="big")
    
    return FingerprintTemplate(
        identifier=identifier,
        image_path=Path(f"{identifier}__fused"),
        protected=protected,
        bit_length=hasher.bit_length,
        quality=avg_quality,
        raw_features=fused_vector,
        minutiae=fused_minutiae,
        fused=True,
        source_count=len(templates),
        consensus_score=consensus_avg,
    )


def fuse_identity_templates(
    templates: Sequence[FingerprintTemplate],
    hasher: CancelableHasher,
    settings: FusionSettings,
) -> List[FingerprintTemplate]:
    """Group templates by identifier and fuse each identity's templates.
    
    Args:
        templates: Sequence of all templates
        hasher: CancelableHasher instance
        settings: Fusion configuration
        
    Returns:
        List of fused templates (one per identity)
    """
    if not settings.enabled:
        return list(templates)
    
    grouped: dict[str, List[FingerprintTemplate]] = {}
    for template in templates:
        grouped.setdefault(template.identifier, []).append(template)
    
    result: List[FingerprintTemplate] = []
    for identifier, group in grouped.items():
        if len(group) <= 1:
            result.extend(group)
            continue
        
        fused = create_fused_template(identifier, group, hasher, settings)
        if fused is None:
            result.extend(group)
            continue
        
        result.append(fused)
        if settings.keep_raw:
            result.extend(group)
    
    return result


# ---------------------------------------------------------------------------
# Quality Assessment


def evaluate_quality(
    minutiae: Sequence[Minutia],
    mask: np.ndarray,
    orientation: np.ndarray,
    frequency: np.ndarray,
    skeleton: np.ndarray,
    enhanced: np.ndarray,
    diffused: np.ndarray,
    *,
    level3: Optional[Sequence[Pore]] = None,
) -> float:
    """Estimate fingerprint quality score in [0, 1].
    
    Combines multiple metrics:
    - Minutiae count/density
    - Foreground area ratio
    - Skeleton quality
    - Orientation/frequency validity
    - Contrast (enhanced and diffused)
    - Level-3 features (if available)
    
    Returns:
        Quality score in [0, 1]
    """
    if mask.size == 0:
        return 0.0
    
    mask_bool = mask.astype(bool)
    foreground_ratio = float(mask_bool.mean()) if mask_bool.size else 0.0
    skeleton_ratio = float(np.clip(skeleton.astype(np.float32).mean(), 0.0, 1.0)) if skeleton.size else 0.0
    
    minutiae_ratio = 0.0
    if MAX_MINUTIAE_ENCODER > 0:
        minutiae_ratio = min(len(minutiae), MAX_MINUTIAE_ENCODER) / float(MAX_MINUTIAE_ENCODER)
    
    orientation_valid = 0.0
    if orientation.size:
        valid_mask = np.isfinite(orientation) & (np.abs(orientation) > 1e-3)
        orientation_valid = float(np.count_nonzero(valid_mask)) / float(orientation.size)
    
    freq_valid = 0.0
    if frequency.size:
        valid_freq = np.isfinite(frequency) & (frequency > 0.0)
        freq_valid = float(np.count_nonzero(valid_freq)) / float(frequency.size)
    
    contrast = 0.0
    diffused_contrast = 0.0
    if mask_bool.any():
        roi = enhanced[mask_bool]
        contrast = float(np.clip((np.std(roi) if roi.size else 0.0) / 64.0, 0.0, 1.0))
        
        diff_roi = diffused[mask_bool]
        diffused_contrast = float(np.clip((np.std(diff_roi) if diff_roi.size else 0.0) / 64.0, 0.0, 1.0))
    
    level3_density = 0.0
    if level3:
        level3_density = min(len(level3) / 25.0, 1.0)
    
    score = (
        0.23 * minutiae_ratio +
        0.20 * foreground_ratio +
        0.14 * skeleton_ratio +
        0.14 * orientation_valid +
        0.10 * freq_valid +
        0.09 * contrast +
        0.05 * diffused_contrast +
        0.05 * level3_density
    )
    
    return float(np.clip(score, 0.0, 1.0))


def build_feature_vector(
    minutiae: Sequence[Minutia],
    mask: np.ndarray,
    orientation: np.ndarray,
    frequency: np.ndarray,
    skeleton: np.ndarray,
    enhanced: np.ndarray,
    diffused: np.ndarray,
    *,
    level3: Optional[Sequence[Pore]] = None,
    use_consensus_weighting: bool = True,
) -> np.ndarray:
    """Build feature vector from fingerprint processing outputs.
    
    Feature components (total 736 dimensions):
    1. Minutiae features (TOP-130): position, angle, quality × 130 minutiae = 650 dims
    2. Global statistics: minutiae count, mask coverage, skeleton density
    3. Minutiae statistics: quality, spatial distribution
    4. Orientation histogram + statistics
    5. Frequency histogram + statistics
    6. Level-3 features (if available)
    7. Intensity statistics (enhanced + diffused)
    8. Gabor filter responses
    
    Args:
        minutiae: List of extracted minutiae
        mask, orientation, frequency, skeleton, enhanced, diffused: Processing outputs
        level3: Optional pore features
        use_consensus_weighting: If True, weight minutiae by consensus (for fused templates)
        
    Returns:
        Feature vector (numpy array, length FEATURE_VECTOR_DIM=736)
    """
    if mask.ndim != 2:
        raise ValueError("Mask must be 2D array")
    
    height, width = mask.shape
    width = max(width, 1)
    height = max(height, 1)
    mask_bool = mask.astype(bool)
    features: List[float] = []
    
    sorted_minutiae = sorted(minutiae, key=lambda m: m.quality, reverse=True)
    
    # 1. Minutiae features (TOP-130)
    for idx in range(MAX_MINUTIAE_ENCODER):
        if idx < len(sorted_minutiae):
            minutia = sorted_minutiae[idx]
            norm_quality = float(np.clip(minutia.quality, 0.0, 255.0)) / 255.0
            
            if use_consensus_weighting:
                consensus_weight = 1.0
                if norm_quality > 0.5:
                    consensus_weight = 1.0 + (norm_quality - 0.5) * 2.0
                elif norm_quality < 0.3:
                    consensus_weight = 0.5 + norm_quality * 1.67
                
                weighted_x = float(minutia.x) / float(width) * consensus_weight
                weighted_y = float(minutia.y) / float(height) * consensus_weight
                weighted_cos = math.cos(minutia.angle) * consensus_weight
                weighted_sin = math.sin(minutia.angle) * consensus_weight
            else:
                weighted_x = float(minutia.x) / float(width)
                weighted_y = float(minutia.y) / float(height)
                weighted_cos = math.cos(minutia.angle)
                weighted_sin = math.sin(minutia.angle)
            
            features.extend([weighted_x, weighted_y, weighted_cos, weighted_sin, norm_quality])
        else:
            features.extend([0.0] * MINUTIA_FEATURE_SIZE)
    
    # 2. Global statistics
    minutiae_count = len(sorted_minutiae)
    features.append(minutiae_count / float(MAX_MINUTIAE_ENCODER or 1))
    features.append(float(mask.mean()))
    features.append(float(skeleton.mean()))
    
    # 3. Minutiae statistics
    if sorted_minutiae:
        qualities = np.array([m.quality for m in sorted_minutiae[:MAX_MINUTIAE_ENCODER]], dtype=np.float32) / 255.0
        xs = np.array([m.x for m in sorted_minutiae[:MAX_MINUTIAE_ENCODER]], dtype=np.float32) / float(width)
        ys = np.array([m.y for m in sorted_minutiae[:MAX_MINUTIAE_ENCODER]], dtype=np.float32) / float(height)
        features.extend([
            float(qualities.mean()), float(qualities.std()),
            float(xs.mean()), float(xs.std()),
            float(ys.mean()), float(ys.std()),
            float(xs.max() - xs.min()), float(ys.max() - ys.min())
        ])
    else:
        features.extend([0.0] * 8)
    
    # 4. Orientation histogram
    orientation_values = orientation.astype(np.float32).flatten()
    orientation_values = orientation_values[np.isfinite(orientation_values)]
    if orientation_values.size:
        hist_ori, _ = np.histogram(orientation_values, bins=ORI_HIST_BINS, range=(-math.pi, math.pi), density=True)
        mean_cos = float(np.mean(np.cos(orientation_values)))
        mean_sin = float(np.mean(np.sin(orientation_values)))
        coherence = math.sqrt(mean_cos ** 2 + mean_sin ** 2)
        std = float(np.std(orientation_values))
        coverage = float(np.count_nonzero(np.abs(orientation_values)) / orientation_values.size)
    else:
        hist_ori = np.zeros(ORI_HIST_BINS, dtype=np.float32)
        mean_cos = mean_sin = coherence = std = coverage = 0.0
    
    features.extend(hist_ori.astype(np.float32))
    features.extend([mean_cos, mean_sin, coherence, std, coverage])
    
    # 5. Frequency histogram
    frequency_values = frequency.astype(np.float32).flatten()
    valid_freq = frequency_values[(frequency_values > 0.0) & np.isfinite(frequency_values)]
    if valid_freq.size:
        upper = float(max(valid_freq.max(), 0.1))
        hist_freq, _ = np.histogram(valid_freq, bins=FREQ_HIST_BINS, range=(0.0, upper), density=True)
        freq_mean = float(valid_freq.mean())
        freq_std = float(valid_freq.std())
        freq_coverage = float(np.count_nonzero(valid_freq) / valid_freq.size)
    else:
        hist_freq = np.zeros(FREQ_HIST_BINS, dtype=np.float32)
        freq_mean = freq_std = freq_coverage = 0.0
    
    features.extend(hist_freq.astype(np.float32))
    features.extend([freq_mean, freq_std, freq_coverage])
    
    # 6. Level-3 features
    if level3:
        radii = np.array([pore.radius for pore in level3], dtype=np.float32)
        strengths = np.array([pore.strength for pore in level3], dtype=np.float32)
        features.extend([
            float(radii.mean()) if radii.size else 0.0,
            float(radii.std()) if radii.size else 0.0,
            float(strengths.mean()) if strengths.size else 0.0,
            float(len(level3))
        ])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])
    
    # 7. Intensity statistics
    foreground = enhanced[mask_bool]
    diffused_foreground = diffused[mask_bool]
    if foreground.size:
        features.extend([float(foreground.mean()) / 255.0, float(foreground.std()) / 255.0])
    else:
        features.extend([0.0, 0.0])
    
    if diffused_foreground.size:
        features.extend([float(diffused_foreground.mean()) / 255.0, float(diffused_foreground.std()) / 255.0])
    else:
        features.extend([0.0, 0.0])
    
    # 8. Gabor filter responses
    for kernel in GABOR_KERNELS:
        response = cv2.filter2D(enhanced, cv2.CV_32F, kernel)
        masked = np.abs(response)[mask_bool]
        if masked.size:
            features.extend([float(masked.mean()), float(masked.std())])
        else:
            features.extend([0.0, 0.0])
    
    # Normalize to FEATURE_VECTOR_DIM
    vector = np.asarray(features, dtype=np.float32)
    if vector.size < FEATURE_VECTOR_DIM:
        vector = np.pad(vector, (0, FEATURE_VECTOR_DIM - vector.size), mode="constant")
    else:
        vector = vector[:FEATURE_VECTOR_DIM]
    
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm
    
    return vector.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Template Persistence


def template_output_path(output_dir: Path, image_path: Path) -> Path:
    """Generate output path for template file."""
    return output_dir / f"{image_path.stem}{TEMPLATE_EXTENSION}"


def save_template(template: FingerprintTemplate, output_dir: Path, *, overwrite: bool = False) -> Path:
    """Save template to disk using pickle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target = template_output_path(output_dir, template.image_path)
    
    if target.exists() and not overwrite:
        return target
    
    with target.open("wb") as handle:
        pickle.dump(template, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return target


def load_templates_from_directory(
    directory: Path,
    *,
    prefer_fused: bool = True,
    include_raw_when_fused: bool = False,
) -> List[FingerprintTemplate]:
    """Load templates from directory.
    
    Args:
        directory: Directory containing .fpt files
        prefer_fused: If True, prefer fused templates over raw
        include_raw_when_fused: If True, include raw templates even when fused exist
        
    Returns:
        List of FingerprintTemplate objects
    """
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Template directory not found: {directory}")
    
    templates: List[FingerprintTemplate] = []
    for file_path in sorted(directory.glob(f"*{TEMPLATE_EXTENSION}")):
        try:
            with file_path.open("rb") as handle:
                loaded = pickle.load(handle)
        except Exception as exc:
            print(f"[warning] Failed to load {file_path.name}: {exc}")
            continue
        
        if isinstance(loaded, FingerprintTemplate):
            templates.append(loaded)
        else:
            print(f"[warning] Ignoring {file_path.name}: incompatible format")
    
    if not templates:
        return templates
    
    # Filter by fusion preference
    if prefer_fused:
        grouped: dict[str, List[FingerprintTemplate]] = {}
        for template in templates:
            grouped.setdefault(template.identifier, []).append(template)
        
        filtered: List[FingerprintTemplate] = []
        for identifier, group in grouped.items():
            fused = [t for t in group if t.fused]
            raw = [t for t in group if not t.fused]
            
            if fused:
                filtered.extend(fused)
                if include_raw_when_fused:
                    filtered.extend(raw)
            else:
                filtered.extend(raw)
        
        return filtered
    
    return templates
