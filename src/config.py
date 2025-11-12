"""Configuration file for Supermatcher v1.0 (Hybrid)

This module contains all configurable parameters for the fingerprint matching system.

Modify these values to tune the system behavior without changing the core code.
"""

from dataclasses import dataclass
from typing import List
import cv2

# ============================================================================
# HARDWARE CONFIGURATION
# ============================================================================

# CPU affinity for P-cores (adjust based on your CPU architecture)
# Example: For Intel 12th gen with 8 P-cores + 8 E-cores, use range(0, 16) for P-cores only
PCORES: List[int] = list(range(0, 12))  # Default: first 12 cores (P-cores)
PCORES_AFFINITY = PCORES  # Alias for compatibility

# ============================================================================
# FILE PATHS AND EXTENSIONS
# ============================================================================

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATE_PATH = SCRIPT_DIR / "templates"  # Default templates directory
IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}  # Supported image formats

# ============================================================================
# QUALITY THRESHOLDS
# ============================================================================

# Minimum quality score for templates (0.0-1.0)
# Lower = more permissive, higher = more strict
QUALITY_THRESHOLD: float = 0.30  # Production default from benchmark
DEFAULT_QUALITY_THRESHOLD = QUALITY_THRESHOLD  # Alias

# ============================================================================
# HASHER CONFIGURATION (CancelableHasher)
# ============================================================================

# Feature vector dimension (input to hasher)
HASHER_FEATURE_DIM: int = 736

# Projection dimension (LSH output size)
HASHER_PROJECTION_DIM: int = 512

# Hasher key (must match existing templates if loading pre-built ones!)
HASHER_KEY: str = "default"

# Number of hash functions (must match existing templates!)
HASHER_HASH_COUNT: int = 2  # v0.5.1 default

# Aliases for backward compatibility
HASH_FEATURE_DIM = HASHER_FEATURE_DIM
HASH_PROJECTION_DIM = HASHER_PROJECTION_DIM
HASH_KEY = HASHER_KEY
HASH_COUNT = HASHER_HASH_COUNT

# ============================================================================
# FUSION SETTINGS
# ============================================================================

@dataclass
class FusionConfig:
    """Configuration for template fusion."""
    enabled: bool = True
    distance: float = 12.0  # Spatial threshold for minutiae consensus (pixels)
    angle_deg: float = 15.0  # Angular threshold for minutiae consensus (degrees)
    min_consensus: float = 0.5  # Minimum consensus ratio (0.0-1.0)
    keep_raw: bool = False  # Keep raw templates after fusion
    mode: str = "optimal"  # Fusion mode: "optimal" or other

# Default fusion settings
DEFAULT_FUSION = FusionConfig()

# Fusion algorithm parameters (v0.5.1 compatibility)
FUSION_MIN_QUALITY: float = 0.3  # Minimum quality threshold for fusion
FUSION_ALIGNMENT_THRESHOLD: float = 15.0  # Alignment threshold (pixels)
FUSION_MINUTIAE_CLUSTER_RADIUS: float = 8.0  # Clustering radius (pixels)
FUSION_ANGLE_TOLERANCE: float = 0.3490658504  # math.radians(20.0)
FUSION_RANSAC_ITERATIONS: int = 1000  # RANSAC max iterations
FUSION_RANSAC_THRESHOLD: float = 10.0  # RANSAC inlier threshold (pixels)
DEFAULT_FUSION_MIN_CONSENSUS: float = 0.4  # Default minimum consensus ratio

# ============================================================================
# MATCHING CONFIGURATION
# ============================================================================

# Hash vs Geometric weighting (must sum to 1.0)
# Optimized via mathematical optimization (Differential Evolution + L-BFGS-B)
# Dataset: 80 probes vs 10 fused templates (benchmark_v1_0)
# Result: Rank-1: 78.75%, Rank-5: 90.00%
HASH_WEIGHT: float = 0.9269  # Hash matching weight (optimized from 0.6)
GEOMETRIC_WEIGHT: float = 0.0731  # Geometric matching weight (optimized from 0.4)

# Geometric matching thresholds (RANSAC)
MATCH_DISTANCE_THRESHOLD: float = 12.0  # Spatial threshold for minutiae correspondence (pixels)
MATCH_ANGLE_THRESHOLD_DEG: float = 15.0  # Angular threshold (degrees)
MATCH_ANGLE_THRESHOLD_RAD: float = 0.2617  # Angular threshold (radians, ~15 degrees)

# RANSAC parameters
RANSAC_SPATIAL_THRESHOLD: float = 18.0  # Spatial inlier threshold (pixels)
RANSAC_ANGULAR_THRESHOLD_DEG: float = 25.0  # Angular inlier threshold (degrees)
RANSAC_MIN_INLIERS: int = 4  # Minimum inliers for valid transformation

# ============================================================================
# ADAPTIVE THRESHOLDS (Verification)
# ============================================================================

# Base threshold for verification matching
BASE_VERIFICATION_THRESHOLD: float = 0.82  # Production value from benchmark (was 0.83, lowered to 0.82)

# Adaptive threshold adjustments based on quality
# High quality (>0.7): use base_threshold
# Medium quality (0.5-0.7): base_threshold - 0.02
# Low quality (<0.5): base_threshold - 0.05
ADAPTIVE_THRESHOLD_HIGH_QUALITY: float = 0.7  # Quality threshold for high tier
ADAPTIVE_THRESHOLD_MEDIUM_QUALITY: float = 0.5  # Quality threshold for medium tier
ADAPTIVE_THRESHOLD_HIGH_ADJUSTMENT: float = 0.0  # No adjustment for high quality
ADAPTIVE_THRESHOLD_MEDIUM_ADJUSTMENT: float = -0.02  # Slight relaxation for medium
ADAPTIVE_THRESHOLD_LOW_ADJUSTMENT: float = -0.05  # More relaxation for low quality

# Identification threshold (1:N matching)
# NOTE: In identification, we use ranking, so threshold is less critical
IDENTIFICATION_THRESHOLD: float = 0.70  # Minimum score to consider a match
MATCH_MIN_SCORE: float = 0.82  # Minimum match score for verification (v0.7 default)

# Tie detection threshold (for geometric reranking activation)
# Optimized via mathematical optimization (Differential Evolution + L-BFGS-B)
# If top 2 scores differ by less than this, use geometric reranking
TIE_THRESHOLD: float = 0.0296  # Optimized from 0.0050
GEOMETRIC_RERANKING_TIE_THRESHOLD: float = 0.0296  # Alias for TIE_THRESHOLD
GEOMETRIC_RERANKING_SCORE_DIFF_THRESHOLD: float = 0.03  # Score difference threshold for reranking

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

# Normalization parameters
NORMALISE_MEAN: float = 100.0  # Target mean value
NORMALISE_STD: float = 20.0  # Target standard deviation

# Segmentation parameters
SEGMENTATION_BLOCK_SIZE: int = 16  # Block size for variance-based segmentation
SEGMENTATION_VARIANCE_THRESHOLD: float = 70.0  # Variance threshold for foreground detection (v0.5.1 = 70.0)

# Coherence diffusion parameters
COHERENCE_ITERATIONS: int = 12  # Number of diffusion iterations (v0.5.1 compatibility)
COHERENCE_KAPPA: float = 5.0  # Diffusion strength
COHERENCE_GAMMA: float = 0.05  # Edge sensitivity

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

# Orientation estimation
ORIENTATION_BLOCK_SIZE: int = 16  # Block size for orientation estimation
ORIENTATION_SMOOTH_SIGMA: float = 7.0  # Gaussian smoothing for orientation field

# Frequency estimation
FREQUENCY_BLOCK_SIZE: int = 32  # Block size for frequency estimation
FREQUENCY_MIN: float = 1.0 / 25.0  # Minimum ridge frequency (1/wavelength)
FREQUENCY_MAX: float = 1.0 / 3.0  # Maximum ridge frequency

# Log-Gabor filter parameters
LOG_GABOR_KERNEL_SIZE: int = 65  # Kernel size (must be odd)
LOG_GABOR_SIGMA_ONEF: float = 0.65  # Frequency bandwidth

# Binarization
import cv2 as _cv2
BINARISATION_METHOD: int = _cv2.THRESH_BINARY + _cv2.THRESH_OTSU  # Automatic threshold with Otsu's method

# Thinning
THINNING_MAX_ITER: int = 40  # Maximum iterations for Zhang-Suen thinning

# Minutiae extraction
MIN_MINUTIAE_QUALITY: float = 0.3  # Minimum quality for a valid minutia
MAX_MINUTIAE_ENCODER: int = 120  # Maximum minutiae to encode in feature vector (v0.5.1 compatibility!)
MINUTIAE_BORDER_MARGIN: int = 8  # Minimum distance from border (pixels) - v0.5.1 compatibility
MINUTIA_PAIR_DISTANCE: float = 12.0  # Minimum distance between minutiae of same type (v0.5.1)

# Feature vector dimensions (v0.5.1 compatibility)
FEATURE_VECTOR_DIM: int = 736  # Total feature vector dimension
MINUTIA_FEATURE_SIZE: int = 5  # (x, y, angle, quality, type) per minutia
ORI_HIST_BINS: int = 32  # Orientation histogram bins
FREQ_HIST_BINS: int = 24  # Frequency histogram bins

# Gabor kernel configuration for texture extraction
GABOR_KERNEL_CONFIG = (
    {"ksize": 21, "sigma": 3.0, "theta": 0.0, "lambd": 7.0, "gamma": 0.6},
    {"ksize": 21, "sigma": 3.0, "theta": 3.14159265359 / 4.0, "lambd": 7.0, "gamma": 0.6},
    {"ksize": 21, "sigma": 3.0, "theta": 3.14159265359 / 2.0, "lambd": 7.0, "gamma": 0.6},
    {"ksize": 21, "sigma": 3.0, "theta": 3.0 * 3.14159265359 / 4.0, "lambd": 7.0, "gamma": 0.6},
    {"ksize": 31, "sigma": 4.5, "theta": 0.0, "lambd": 11.0, "gamma": 0.5},
    {"ksize": 31, "sigma": 4.5, "theta": 3.14159265359 / 4.0, "lambd": 11.0, "gamma": 0.5},
    {"ksize": 31, "sigma": 4.5, "theta": 3.14159265359 / 2.0, "lambd": 11.0, "gamma": 0.5},
    {"ksize": 31, "sigma": 4.5, "theta": 3.0 * 3.14159265359 / 4.0, "lambd": 11.0, "gamma": 0.5},
)

# Pre-computed Gabor kernels (generated from GABOR_KERNEL_CONFIG)
GABOR_KERNELS = tuple(
    cv2.getGaborKernel(
        (cfg["ksize"], cfg["ksize"]),
        cfg["sigma"],
        cfg["theta"],
        cfg["lambd"],
        cfg["gamma"],
        psi=0.0,
        ktype=cv2.CV_32F,
    )
    for cfg in GABOR_KERNEL_CONFIG
)

# Level-3 features (pores) - if enabled
PORE_MIN_RADIUS: float = 1.0  # Minimum pore radius (pixels)
PORE_MAX_RADIUS: float = 4.0  # Maximum pore radius (pixels)
PORE_MIN_STRENGTH: float = 0.3  # Minimum detection strength

# ============================================================================
# TEMPLATE STORAGE
# ============================================================================

# Template file extension
TEMPLATE_EXTENSION: str = ".fpt"  # Fingerprint template (pickle format)

# Template naming convention for fused templates
FUSED_TEMPLATE_SUFFIX: str = "__fused"  # Suffix for fused master templates

# ============================================================================
# FEATURE VECTOR CONSTRUCTION
# ============================================================================

# Consensus weighting for feature vectors (used in fusion)
USE_CONSENSUS_WEIGHTING: bool = True

# Feature vector components and weights (used in quality assessment)
QUALITY_WEIGHT_MINUTIAE: float = 0.23
QUALITY_WEIGHT_FOREGROUND: float = 0.20
QUALITY_WEIGHT_SKELETON: float = 0.14
QUALITY_WEIGHT_ORIENTATION: float = 0.14
QUALITY_WEIGHT_FREQUENCY: float = 0.10
QUALITY_WEIGHT_CONTRAST: float = 0.09
QUALITY_WEIGHT_DIFFUSED: float = 0.05
QUALITY_WEIGHT_LEVEL3: float = 0.05  # If level-3 features enabled

# ============================================================================
# LOGGING AND DEBUG
# ============================================================================

# Silent mode (suppress verbose geometric reranking logs)
SILENT_MODE: bool = False  # Set to True to suppress logs

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration consistency."""
    errors = []
    
    # Check weight sum
    weight_sum = HASH_WEIGHT + GEOMETRIC_WEIGHT
    if not (0.99 <= weight_sum <= 1.01):
        errors.append(f"HASH_WEIGHT + GEOMETRIC_WEIGHT must sum to 1.0 (got {weight_sum})")
    
    # Check thresholds
    if not (0.0 <= QUALITY_THRESHOLD <= 1.0):
        errors.append(f"QUALITY_THRESHOLD must be in [0.0, 1.0] (got {QUALITY_THRESHOLD})")
    
    if not (0.0 <= BASE_VERIFICATION_THRESHOLD <= 1.0):
        errors.append(f"BASE_VERIFICATION_THRESHOLD must be in [0.0, 1.0] (got {BASE_VERIFICATION_THRESHOLD})")
    
    # Check hasher config
    if HASHER_FEATURE_DIM <= 0:
        errors.append(f"HASHER_FEATURE_DIM must be positive (got {HASHER_FEATURE_DIM})")
    
    if HASHER_PROJECTION_DIM <= 0:
        errors.append(f"HASHER_PROJECTION_DIM must be positive (got {HASHER_PROJECTION_DIM})")
    
    if HASHER_HASH_COUNT <= 0:
        errors.append(f"HASHER_HASH_COUNT must be positive (got {HASHER_HASH_COUNT})")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))

# Run validation on import
validate_config()
