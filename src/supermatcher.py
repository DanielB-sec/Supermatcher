"""Hybrid fingerprint identification/authentication pipeline.

This module implements a multi-stage algorithm tailored for challenging datasets
such as FVC 2002 DB3.  The pipeline follows four explicit phases:

1. Pre-processing and segmentation using local normalisation and block variance.
2. Robust enhancement leveraging coherence-enhancing diffusion and Log-Gabor filters.
3. Minutiae extraction via binarisation, thinning, crossing-number analysis and
   post-processing heuristics to eliminate spurious minutiae.
4. (Optional) Level-3 feature fusion for high-resolution samples, including
   pore-based descriptors combined through Delaunay triangulation.

The script exposes a CLI that can perform identification (one-to-many) and
authentication (one-to-one) against a fingerprint gallery stored inside the
``fingerprints`` folder located next to this file.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial import KDTree


# ---------------------------------------------------------------------------
# Data structures


@dataclass
class Minutia:
	x: float
	y: float
	angle: float  # radians, ridge direction
	kind: str     # "ending" or "bifurcation"
	quality: float


@dataclass
class Pore:
	x: float
	y: float
	radius: float
	strength: float


@dataclass
class FingerprintTemplate:
	identifier: str
	image_path: Path
	protected: np.ndarray  # packed cancelable template bits
	bit_length: int
	quality: float = 0.0
	raw_features: Optional[np.ndarray] = None
	minutiae: Optional[List[Minutia]] = None
	fused: bool = False
	source_count: int = 1
	consensus_score: float = 1.0

	def __post_init__(self) -> None:
		self.protected = np.asarray(self.protected, dtype=np.uint8)
		if self.protected.ndim != 1:
			raise ValueError("Protected template must be a 1-D array of packed bits.")
		expected_len = (self.bit_length + 7) // 8
		if self.protected.size != expected_len:
			raise ValueError(
				f"Protected template has {self.protected.size} bytes but expected {expected_len}."
			)
		self.quality = float(self.quality)
		if not math.isfinite(self.quality):
			self.quality = 0.0
		self.quality = float(np.clip(self.quality, 0.0, 1.0))
		if self.raw_features is not None:
			raw = np.asarray(self.raw_features, dtype=np.float32)
			if raw.ndim != 1:
				raise ValueError("raw_features must be a 1-D vector if provided.")
			self.raw_features = raw
		if self.minutiae is not None:
			self.minutiae = [
				m if isinstance(m, Minutia) else Minutia(**m)  # type: ignore[arg-type]
				for m in self.minutiae
			]
		self.fused = bool(self.fused)
		self.source_count = int(max(self.source_count, 1))
		self.consensus_score = float(np.clip(self.consensus_score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Constants and configuration defaults


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = SCRIPT_DIR / "fingerprints"
DEFAULT_TEMPLATE_PATH = SCRIPT_DIR / "templates"

IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
TEMPLATE_EXTENSION = ".fpt"


# Phase 1 parameters (REVERTED to original)
NORMALISATION_BLOCK = 16  # REVERTED from 8 back to 16
NORMALISATION_BLOCK_SECONDARY = 16  # For dual-block mode
NORMALISATION_MEAN = 100.0
NORMALISATION_VAR = 100.0

SEGMENT_BLOCK = 16
SEGMENT_THRESHOLD = 70.0


# Phase 2 parameters (REVERTED to original)
CED_ITERATIONS = 12  # REVERTED from 15 back to 12
CED_ITERATIONS_MIN = 12  # Same as base
CED_ITERATIONS_MAX = 14  # Small range only for worst cases
CED_DELTA_T = 0.15
CED_TENSOR_SIGMA = 2.0
CED_GRAD_SIGMA = 1.0
CED_ALPHA = 0.01
CED_BETA = 1.25

ORIENTATION_BLOCK = 16
LOG_GABOR_SCALES = (0.85, 1.0, 1.2)
LOG_GABOR_SIGMA_R = 1.5
LOG_GABOR_SIGMA_THETA_DEG = 12.0


# Phase 3 parameters (REVERTED to original)
BINARISATION_METHOD = cv2.THRESH_BINARY + cv2.THRESH_OTSU
THINNING_MAX_ITER = 40
MINUTIA_DISTANCE_THRESHOLD = 5
MINUTIA_EDGE_MARGIN = 8
MINUTIA_PAIR_DISTANCE = 12.0
MINUTIA_ANGLE_THRESHOLD_DEG = 25.0


# Matching parameters
MATCH_DISTANCE_THRESHOLD = 18.0
MATCH_ANGLE_THRESHOLD_RAD = math.radians(25.0)
MATCH_MIN_SCORE = 0.82

# Level 3 (optional)
PORE_MIN_SIGMA = 0.6
PORE_MAX_SIGMA = 1.4
PORE_SIGMA_STEPS = 5
PORE_RESPONSE_THRESHOLD = 0.025

# Cancelable biometrics parameters
FEATURE_VECTOR_DIM = 736
MAX_MINUTIAE_ENCODER = 120
MINUTIA_FEATURE_SIZE = 5
ORI_HIST_BINS = 32
FREQ_HIST_BINS = 24
DEFAULT_PROJECTION_DIM = 512
DEFAULT_HASH_COUNT = 2
DEFAULT_QUALITY_THRESHOLD = 0.45
DEFAULT_FUSION_DISTANCE = 12.0
DEFAULT_FUSION_ANGLE_DEG = 20.0
DEFAULT_FUSION_MIN_CONSENSUS = 0.4

# Advanced fusion parameters (REVERTED to original)
FUSION_MIN_QUALITY = 0.3  # REVERTED from 0.28 back to 0.3
FUSION_ALIGNMENT_THRESHOLD = 15.0  # pixels
FUSION_MINUTIAE_CLUSTER_RADIUS = 8.0  # REVERTED from 11.0 back to 8.0
FUSION_ANGLE_TOLERANCE = math.radians(20.0)
FUSION_RANSAC_ITERATIONS = 1000
FUSION_RANSAC_THRESHOLD = 10.0
DEFAULT_FUSION_MIN_CONSENSUS = 0.4  # REVERTED from 0.35 back to 0.4


# ---------------------------------------------------------------------------
# Fusion configuration


@dataclass
class FusionSettings:
	enabled: bool = True
	distance: float = DEFAULT_FUSION_DISTANCE
	angle_deg: float = DEFAULT_FUSION_ANGLE_DEG
	min_consensus: float = DEFAULT_FUSION_MIN_CONSENSUS
	keep_raw: bool = False
	mode: str = "full"  # none, minutiae, features, full

	@property
	def angle_rad(self) -> float:
		return math.radians(self.angle_deg)

GABOR_KERNEL_CONFIG = (
	{"ksize": 21, "sigma": 3.0, "theta": 0.0, "lambd": 7.0, "gamma": 0.6},
	{"ksize": 21, "sigma": 3.0, "theta": math.pi / 4.0, "lambd": 7.0, "gamma": 0.6},
	{"ksize": 21, "sigma": 3.0, "theta": math.pi / 2.0, "lambd": 7.0, "gamma": 0.6},
	{"ksize": 21, "sigma": 3.0, "theta": 3.0 * math.pi / 4.0, "lambd": 7.0, "gamma": 0.6},
	{"ksize": 31, "sigma": 4.5, "theta": 0.0, "lambd": 11.0, "gamma": 0.5},
	{"ksize": 31, "sigma": 4.5, "theta": math.pi / 4.0, "lambd": 11.0, "gamma": 0.5},
	{"ksize": 31, "sigma": 4.5, "theta": math.pi / 2.0, "lambd": 11.0, "gamma": 0.5},
	{"ksize": 31, "sigma": 4.5, "theta": 3.0 * math.pi / 4.0, "lambd": 11.0, "gamma": 0.5},
)

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

TEXTURE_FEATURES_PER_KERNEL = 2


# ---------------------------------------------------------------------------
# Utility helpers


def load_grayscale_image(path: Path) -> np.ndarray:
	"""Load fingerprint image - REVERTED to original (no filtering)."""
	image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
	if image is None:
		raise FileNotFoundError(f"Unable to read fingerprint image: {path}")
	return image.astype(np.float32)


def pre_denoise(image: np.ndarray) -> np.ndarray:
	"""Apply bilateral filtering only (SIMPLIFIED).
	
	ROLLBACK: Removed NLMeans denoising as it was too aggressive for low-quality fingerprints.
	Now uses only bilateral filter to preserve ridge details.
	
	Args:
		image: Input grayscale image (float32)
	
	Returns:
		Denoised image maintaining ridge structures
	"""
	# Convert to uint8 for denoising operations
	img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
	
	# Apply only bilateral filter (preserves edges while smoothing)
	# d=5: neighborhood diameter, sigmaColor=50, sigmaSpace=50
	bilateral = cv2.bilateralFilter(img_uint8, d=5, sigmaColor=50, sigmaSpace=50)
	
	return bilateral.astype(np.float32)


def normalise_image(image: np.ndarray,
					block_size: int = NORMALISATION_BLOCK,
					mean0: float = NORMALISATION_MEAN,
					var0: float = NORMALISATION_VAR,
					dual_block: bool = False) -> np.ndarray:
	"""Normalise intensity using local statistics - REVERTED to original single-block."""
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
								block_size: int = SEGMENT_BLOCK,
								threshold: float = SEGMENT_THRESHOLD) -> np.ndarray:
	"""Segment fingerprint foreground using block variance criterion."""
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

	kernel = np.ones((3, 3), np.uint8)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
	return mask.astype(bool)


def gaussian_derivatives(image: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
	"""Compute smoothed derivatives using Gaussian kernels (float64 for stability)."""
	ksize = int(6 * sigma + 1) | 1
	blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)
	gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
	gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
	return gx, gy


def coherence_diffusion(image: np.ndarray,
						mask: np.ndarray,
						iterations: int = CED_ITERATIONS,
						dt: float = CED_DELTA_T,
						grad_sigma: float = CED_GRAD_SIGMA,
						tensor_sigma: float = CED_TENSOR_SIGMA,
						alpha: float = CED_ALPHA,
						beta: float = CED_BETA,
						adaptive_iterations: bool = False) -> np.ndarray:
	"""Apply coherence-enhancing diffusion - REVERTED to original (no adaptation)."""
	I = image.astype(np.float64, copy=True)
	valid = mask.astype(bool)
	
	for _ in range(iterations):
		gx, gy = gaussian_derivatives(I, grad_sigma)
		Jxx = cv2.GaussianBlur(gx * gx, (0, 0), tensor_sigma)
		Jyy = cv2.GaussianBlur(gy * gy, (0, 0), tensor_sigma)
		Jxy = cv2.GaussianBlur(gx * gy, (0, 0), tensor_sigma)

		trace = Jxx + Jyy
		diff = Jxx - Jyy
		discriminant = np.sqrt(np.maximum(diff * diff + 4.0 * Jxy * Jxy, 0.0))

		lambda1 = 0.5 * (trace + discriminant)
		lambda2 = 0.5 * (trace - discriminant)

		denom = 1e-6 + lambda1 * lambda2 + 1e-6
		num = (lambda1 - lambda2) ** 2
		exponent = -np.divide(num, denom, out=np.zeros_like(num), where=denom != 0.0)
		exponent = np.clip(exponent, -60.0, 0.0)
		coherence = np.exp(exponent)
		c_parallel = alpha + (beta - alpha) * (1.0 - coherence)
		c_perp = alpha

		vx = 2.0 * Jxy
		vy = Jyy - Jxx + discriminant
		norm = np.sqrt(vx * vx + vy * vy)
		norm = np.where(norm < 1e-9, 1.0, norm)
		vx = vx / norm
		vy = vy / norm

		Dxx = c_parallel * vx * vx + c_perp * vy * vy
		Dyy = c_parallel * vy * vy + c_perp * vx * vx
		Dxy = (c_parallel - c_perp) * vx * vy

		Ix = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)
		Iy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)

		Jx = Dxx * Ix + Dxy * Iy
		Jy = Dxy * Ix + Dyy * Iy

		divJx = cv2.Sobel(Jx, cv2.CV_64F, 1, 0, ksize=3)
		divJy = cv2.Sobel(Jy, cv2.CV_64F, 0, 1, ksize=3)
		divergence = divJx + divJy

		I[valid] = I[valid] + dt * divergence[valid]

	return np.clip(I, 0.0, 255.0).astype(np.float32)


def estimate_orientation_and_frequency(image: np.ndarray,
									   mask: np.ndarray,
									   block_size: int = ORIENTATION_BLOCK) -> Tuple[np.ndarray, np.ndarray]:
	"""Estimate ridge orientation (radians) and spatial frequency (cycles/pixel)."""
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
				orientation[by, bx] = 0.0
				frequency[by, bx] = 0.0
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
	"""Estimate local ridge frequency from 1D signature orthogonal to orientation."""
	if not np.isfinite(theta):
		return 0.0
	h, w = block.shape
	center = np.array([(w - 1) / 2.0, (h - 1) / 2.0], dtype=np.float32)

	direction = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
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

	if peak_index >= len(magnitudes) or not np.isfinite(peak_index):
		return 0.0

	frequency = peak_index / (length)
	return float(frequency)


def create_log_gabor_kernel(size: int,
							f0: float,
							theta0: float,
							sigma_r: float = LOG_GABOR_SIGMA_R,
							sigma_theta_deg: float = LOG_GABOR_SIGMA_THETA_DEG) -> np.ndarray:
	"""Construct a Log-Gabor filter in the frequency domain for given parameters."""
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
	radial[~valid] = 0.0

	angular = np.exp(-(theta_diff ** 2) / (2 * sigma_theta ** 2))
	filter_kernel = radial * angular
	return filter_kernel.astype(np.complex64)


def apply_log_gabor_enhancement(image: np.ndarray,
								mask: np.ndarray,
								orientation: np.ndarray,
								frequency: np.ndarray,
								block_size: int = ORIENTATION_BLOCK,
								scales: Sequence[float] = LOG_GABOR_SCALES) -> np.ndarray:
	"""Enhance ridges using a bank of Log-Gabor filters per block."""
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
	"""Binarise enhanced image and compute skeleton using Zhang-Suen thinning."""
	clipped = image.copy()
	clipped[~mask] = 0.0
	clipped_uint8 = np.clip(clipped, 0, 255).astype(np.uint8)
	_, binary = cv2.threshold(clipped_uint8, 0, 255, BINARISATION_METHOD)
	binary = (binary > 0).astype(np.uint8)

	skeleton = zhang_suen_thinning(binary)
	skeleton &= mask.astype(np.uint8)
	return binary, skeleton


def zhang_suen_thinning(binary: np.ndarray, max_iter: int = THINNING_MAX_ITER) -> np.ndarray:
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
					P = img[y, x]
					if P != 1:
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
					 block_size: int = ORIENTATION_BLOCK) -> List[Minutia]:
	"""Extract minutiae using the Crossing Number method."""
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

			if min(x, w - 1 - x, y, h - 1 - y) < MINUTIA_EDGE_MARGIN:
				continue

			theta = block_orientation(y, x)
			neighbourhood_var = neighbourhood.var()
			minutiae.append(Minutia(float(x), float(y), theta, kind, float(neighbourhood_var)))

	return minutiae


def validate_minutiae(minutiae: List[Minutia],
					  skeleton: np.ndarray,
					  mask: np.ndarray,
					  validate_with_context: bool = True) -> List[Minutia]:
	"""Remove spurious minutiae based on spatial heuristics.
	
	Args:
		minutiae: List of extracted minutiae
		skeleton: Thinned ridge skeleton
		mask: Foreground mask
		validate_with_context: If True, apply neighborhood and angular consistency checks
	
	Returns:
		Filtered minutiae list
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


def refine_minutiae_geometry(minutiae: List[Minutia],
							  skeleton: np.ndarray,
							  binary: np.ndarray) -> List[Minutia]:
	"""Refine minutiae positions and validate geometry using morphology.
	
	Args:
		minutiae: List of validated minutiae
		skeleton: Thinned ridge skeleton
		binary: Binary fingerprint image
	
	Returns:
		Refined minutiae with adjusted coordinates and validated types
	"""
	if not minutiae:
		return []
	
	h, w = skeleton.shape
	refined: List[Minutia] = []
	
	for m in minutiae:
		x_int, y_int = int(m.x), int(m.y)
		
		# Extract 8-pixel radius neighborhood for geometry validation
		radius = 8
		y_min = max(0, y_int - radius)
		y_max = min(h, y_int + radius + 1)
		x_min = max(0, x_int - radius)
		x_max = min(w, x_int + radius + 1)
		
		skeleton_patch = skeleton[y_min:y_max, x_min:x_max]
		
		# Validate minutia type using morphological analysis
		if m.kind == "ending":
			# For endings: verify ridge truly terminates (dilation check)
			dilated = cv2.dilate(skeleton_patch, np.ones((3, 3), np.uint8), iterations=1)
			# Ending should have limited connectivity
			if dilated.sum() > skeleton_patch.sum() * 2.5:
				# Too much connectivity after dilation, might be false ending
				continue
		
		elif m.kind == "bifurcation":
			# For bifurcations: confirm 3 distinct branches exist
			# Count connected components in neighborhood
			coords = np.argwhere(skeleton_patch > 0)
			if len(coords) < 3:
				# Insufficient ridge pixels for bifurcation
				continue
		
		# Adjust coordinates to local center of mass (max 3 pixels displacement)
		skeleton_coords = np.argwhere(skeleton_patch > 0)
		if len(skeleton_coords) > 0:
			# Compute center of mass
			cy, cx = skeleton_coords.mean(axis=0)
			
			# Convert to global coordinates
			global_cy = y_min + cy
			global_cx = x_min + cx
			
			# Check displacement
			displacement = math.hypot(global_cx - m.x, global_cy - m.y)
			
			if displacement < 3.0:
				# Accept adjusted position
				refined.append(Minutia(
					x=float(global_cx),
					y=float(global_cy),
					angle=m.angle,
					kind=m.kind,
					quality=m.quality,
				))
			else:
				# Displacement too large, keep original
				refined.append(m)
		else:
			# No skeleton pixels found, keep original
			refined.append(m)
	
	return refined


def detect_pores(image: np.ndarray,
				 mask: np.ndarray,
				 min_sigma: float = PORE_MIN_SIGMA,
				 max_sigma: float = PORE_MAX_SIGMA,
				 steps: int = PORE_SIGMA_STEPS,
				 threshold: float = PORE_RESPONSE_THRESHOLD) -> List[Pore]:
	"""Detect pores using a Laplacian-of-Gaussian multi-scale response."""
	pores: List[Pore] = []
	sigmas = np.linspace(min_sigma, max_sigma, steps)
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


def compute_delaunay_signatures(minutiae: Sequence[Minutia], image_shape: Tuple[int, int]) -> List[Tuple[float, float, float]]:
	"""Compute simple Delaunay-based triangle signatures for fusion stage."""
	if len(minutiae) < 3:
		return []

	subdiv = cv2.Subdiv2D((0, 0, image_shape[1], image_shape[0]))
	for m in minutiae:
		subdiv.insert((m.x, m.y))

	triangles = subdiv.getTriangleList()
	signatures: List[Tuple[float, float, float]] = []
	for tri in triangles:
		x1, y1, x2, y2, x3, y3 = tri
		if not (0 <= x1 < image_shape[1] and 0 <= x2 < image_shape[1] and 0 <= x3 < image_shape[1]):
			continue
		if not (0 <= y1 < image_shape[0] and 0 <= y2 < image_shape[0] and 0 <= y3 < image_shape[0]):
			continue
		a = float(math.hypot(x1 - x2, y1 - y2))
		b = float(math.hypot(x2 - x3, y2 - y3))
		c = float(math.hypot(x3 - x1, y3 - y1))
		perimeter = a + b + c
		if perimeter <= 0:
			continue
		lengths = sorted([a / perimeter, b / perimeter, c / perimeter])
		lengths_tuple = (float(lengths[0]), float(lengths[1]), float(lengths[2]))
		signatures.append(lengths_tuple)
	return signatures


# ---------------------------------------------------------------------------
# Cancelable template helpers


def _derive_seed_from_key(key: str, feature_dim: int, projection_dim: int) -> np.random.Generator:
	key_material = f"{key}|{feature_dim}|{projection_dim}".encode("utf-8")
	digest = hashlib.sha256(key_material).digest()
	seed = int.from_bytes(digest[:8], "big", signed=False)
	return np.random.default_rng(seed)


class CancelableHasher:
	def __init__(
		self,
		feature_dim: int,
		projection_dim: int,
		key: str,
		hash_count: int = 1,
	) -> None:
		if projection_dim <= 0:
			raise ValueError("projection_dim must be positive.")
		if hash_count <= 0:
			raise ValueError("hash_count must be positive.")
		self.feature_dim = feature_dim
		self.projection_dim = projection_dim
		self.hash_count = hash_count
		self._projections: List[np.ndarray] = []
		self._biases: List[np.ndarray] = []
		for index in range(hash_count):
			seed_key = f"{key}:{index}"
			rng = _derive_seed_from_key(seed_key, feature_dim, projection_dim)
			projection = rng.normal(
				loc=0.0,
				scale=1.0 / math.sqrt(projection_dim),
				size=(projection_dim, feature_dim),
			).astype(np.float32)
			bias = rng.normal(loc=0.0, scale=0.05, size=(projection_dim,)).astype(np.float32)
			self._projections.append(projection)
			self._biases.append(bias)
		self._pack_length = (self.bit_length + 7) // 8

	@property
	def bit_length(self) -> int:
		return self.projection_dim * self.hash_count

	def encode(self, features: np.ndarray) -> np.ndarray:
		vector = np.asarray(features, dtype=np.float32)
		if vector.ndim != 1:
			raise ValueError("Feature vector must be one dimensional.")
		if vector.shape[0] != self.feature_dim:
			raise ValueError(
				f"Expected feature vector of length {self.feature_dim}, received {vector.shape[0]}"
			)
		bits_accum = []
		for projection, bias in zip(self._projections, self._biases):
			projected = (projection @ vector) + bias
			bits_accum.append((projected >= 0.0).astype(np.uint8))
		concatenated = np.concatenate(bits_accum, axis=0)
		return np.packbits(concatenated)

	def similarity(self, packed_a: np.ndarray, packed_b: np.ndarray) -> float:
		a = np.asarray(packed_a, dtype=np.uint8)
		b = np.asarray(packed_b, dtype=np.uint8)
		if a.shape != b.shape:
			raise ValueError("Packed templates must have identical shape to compare.")
		if a.size != self._pack_length:
			raise ValueError("Packed template size does not match projection parameters.")
		xor = np.bitwise_xor(a, b)
		mismatches = np.unpackbits(xor)[: self.bit_length]
		if mismatches.size != self.bit_length:
			raise ValueError("Packed template size is inconsistent with projection parameters.")
		reshaped = mismatches.reshape(self.hash_count, self.projection_dim)
		per_hash_similarity = 1.0 - (reshaped.sum(axis=1) / float(self.projection_dim))
		return float(per_hash_similarity.mean())


def _angle_difference(theta1: float, theta2: float) -> float:
	diff = (theta1 - theta2 + math.pi) % (2.0 * math.pi) - math.pi
	return abs(diff)


def fuse_feature_vectors(templates: Sequence[FingerprintTemplate]) -> Optional[np.ndarray]:
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
# Advanced Fusion Helpers


def assess_image_quality(
	image: np.ndarray,
	mask: np.ndarray,
	skeleton: np.ndarray,
	minutiae: Sequence[Minutia],
) -> float:
	"""Assess fingerprint image quality using multiple metrics.
	
	Args:
		image: Enhanced fingerprint image (float32, 0-255)
		mask: Binary foreground mask
		skeleton: Thinned skeleton
		minutiae: List of extracted minutiae
		
	Returns:
		Quality score in range [0, 1]
	"""
	if mask.size == 0 or not mask.any():
		return 0.0
		
	# 1. Sharpness via Laplacian variance
	laplacian = cv2.Laplacian(image.astype(np.uint8), cv2.CV_64F)
	laplacian_var = float(laplacian[mask].var()) if mask.any() else 0.0
	sharpness_score = float(np.clip(laplacian_var / 500.0, 0.0, 1.0))
	
	# 2. Foreground area ratio
	area_score = float(mask.mean())
	
	# 3. Minutiae density and distribution
	minutiae_count = len(minutiae)
	minutiae_score = float(np.clip(minutiae_count / 50.0, 0.0, 1.0))
	
	# 4. Skeleton quality
	skeleton_density = float(skeleton[mask].mean()) if mask.any() else 0.0
	skeleton_score = float(np.clip(skeleton_density * 5.0, 0.0, 1.0))
	
	# 5. Signal-to-noise ratio
	if mask.any():
		foreground = image[mask]
		snr = float(foreground.mean() / (foreground.std() + 1e-6))
		snr_score = float(np.clip(snr / 10.0, 0.0, 1.0))
	else:
		snr_score = 0.0
	
	# Weighted combination
	quality = (
		0.30 * sharpness_score +
		0.20 * area_score +
		0.25 * minutiae_score +
		0.15 * skeleton_score +
		0.10 * snr_score
	)
	
	return float(np.clip(quality, 0.0, 1.0))


def estimate_rigid_transform_ransac(
	src_points: np.ndarray,
	dst_points: np.ndarray,
	max_iterations: int = FUSION_RANSAC_ITERATIONS,
	threshold: float = FUSION_RANSAC_THRESHOLD,
) -> Tuple[Optional[np.ndarray], float]:
	"""Estimate rigid transformation (rotation + translation) using RANSAC.
	
	Args:
		src_points: Source points (N, 2)
		dst_points: Destination points (N, 2)
		max_iterations: Maximum RANSAC iterations
		threshold: Inlier distance threshold
		
	Returns:
		Tuple of (transformation_matrix 2x3, confidence_score)
	"""
	if src_points.shape[0] < 3 or dst_points.shape[0] < 3:
		return None, 0.0
	
	if src_points.shape[0] != dst_points.shape[0]:
		return None, 0.0
	
	best_transform = None
	best_inliers = 0
	n_points = src_points.shape[0]
	
	for _ in range(max_iterations):
		# Randomly sample 2 point pairs
		if n_points < 2:
			break
		indices = np.random.choice(n_points, size=min(2, n_points), replace=False)
		src_sample = src_points[indices]
		dst_sample = dst_points[indices]
		
		# Estimate transformation from 2 pairs
		transform = cv2.estimateAffinePartial2D(
			src_sample.reshape(-1, 1, 2).astype(np.float32),
			dst_sample.reshape(-1, 1, 2).astype(np.float32),
		)
		
		if transform is None or transform[0] is None:
			continue
		
		T = transform[0]
		
		# Apply transformation and count inliers
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
	
	# Extract positions
	ref_points = np.array([[m.x, m.y] for m in reference_minutiae], dtype=np.float32)
	tgt_points = np.array([[m.x, m.y] for m in target_minutiae], dtype=np.float32)
	
	# Find correspondences using nearest neighbors (simplified)
	tree = KDTree(ref_points)
	distances, indices = tree.query(tgt_points, k=1)
	
	# Filter correspondences by distance threshold
	valid = distances < FUSION_ALIGNMENT_THRESHOLD
	if valid.sum() < 3:
		return list(target_minutiae), 0.0
	
	src_matched = tgt_points[valid]
	dst_matched = ref_points[indices[valid]]
	
	# Estimate transformation
	transform, confidence = estimate_rigid_transform_ransac(src_matched, dst_matched)
	
	if transform is None:
		return list(target_minutiae), 0.0
	
	# Apply transformation to all minutiae
	aligned_minutiae: List[Minutia] = []
	for minutia in target_minutiae:
		point = np.array([[minutia.x, minutia.y]], dtype=np.float32).reshape(1, 1, 2)
		transformed = cv2.transform(point, transform).reshape(2)
		
		# Extract rotation from transform matrix
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


def fuse_minutiae_advanced(
	aligned_minutiae_sets: Sequence[Sequence[Minutia]],
	quality_weights: Sequence[float],
	adaptive_consensus: bool = True,
) -> List[Minutia]:
	"""Fuse multiple aligned minutiae sets using improved spatial clustering.
	
	Args:
		aligned_minutiae_sets: List of aligned minutiae lists
		quality_weights: Quality weight for each set
		adaptive_consensus: If True, adjust consensus threshold based on sample quality
		
	Returns:
		Fused minutiae list with consensus filtering
	"""
	if not aligned_minutiae_sets:
		return []
	
	n_samples = len(aligned_minutiae_sets)
	
	# Normalize weights
	weights = np.array(quality_weights, dtype=np.float32)
	if not np.allclose(weights.sum(), 0.0):
		weights /= weights.sum()
	else:
		weights = np.ones_like(weights) / float(len(weights))
	
	# Compute average sample quality for adaptive consensus
	avg_quality = float(weights.mean())
	
	# Adjust consensus threshold adaptively
	if adaptive_consensus:
		min_consensus_adjusted = max(0.25, DEFAULT_FUSION_MIN_CONSENSUS * avg_quality)
		print(f"[fusion] Adaptive consensus threshold: {min_consensus_adjusted:.2f} (avg quality: {avg_quality:.2f})")
	else:
		min_consensus_adjusted = DEFAULT_FUSION_MIN_CONSENSUS
	
	# Sort minutiae by quality before clustering (process best first)
	all_minutiae_with_meta: List[Tuple[Minutia, int, float]] = []
	for sample_idx, minutiae_set in enumerate(aligned_minutiae_sets):
		weight = float(weights[sample_idx])
		for minutia in minutiae_set:
			all_minutiae_with_meta.append((minutia, sample_idx, weight))
	
	# Sort by quality descending
	all_minutiae_with_meta.sort(key=lambda item: item[0].quality * item[2], reverse=True)
	
	clusters: List[dict] = []
	
	# Aggregate minutiae from all sets with improved clustering
	for minutia, sample_idx, weight in all_minutiae_with_meta:
		x, y, angle = float(minutia.x), float(minutia.y), float(minutia.angle)
		local_quality = max(float(minutia.quality), 1e-6)
		combined_weight = weight * local_quality
		
		# Exponentially weighted quality (weight^1.5 for emphasis on high-quality samples)
		combined_weight = combined_weight ** 1.5
		
		# Find matching cluster using weighted distance
		assigned = False
		for cluster in clusters:
			spatial_dist = math.hypot(x - cluster["centroid_x"], y - cluster["centroid_y"])
			if spatial_dist > FUSION_MINUTIAE_CLUSTER_RADIUS:
				continue
			
			angle_diff = _angle_difference(angle, cluster["mean_angle"])
			if angle_diff > FUSION_ANGLE_TOLERANCE:
				continue
			
			# Weighted distance combining spatial and angular components
			weighted_dist = spatial_dist * (1.0 + 0.3 * angle_diff / FUSION_ANGLE_TOLERANCE)
			
			if weighted_dist > FUSION_MINUTIAE_CLUSTER_RADIUS:
				continue
			
			# Add to cluster
			cluster["sum_weights"] += combined_weight
			cluster["sum_x"] += combined_weight * x
			cluster["sum_y"] += combined_weight * y
			cluster["sum_cos"] += combined_weight * math.cos(angle)
			cluster["sum_sin"] += combined_weight * math.sin(angle)
			cluster["sample_ids"].add(sample_idx)
			cluster["kind_weights"][minutia.kind] = cluster["kind_weights"].get(minutia.kind, 0.0) + combined_weight
			
			# Update centroid
			cluster["centroid_x"] = cluster["sum_x"] / cluster["sum_weights"]
			cluster["centroid_y"] = cluster["sum_y"] / cluster["sum_weights"]
			cluster["mean_angle"] = math.atan2(cluster["sum_sin"], cluster["sum_cos"])
			
			assigned = True
			break
		
		if not assigned:
			# Create new cluster
			clusters.append({
				"sum_weights": combined_weight,
				"sum_x": combined_weight * x,
				"sum_y": combined_weight * y,
				"sum_cos": combined_weight * math.cos(angle),
				"sum_sin": combined_weight * math.sin(angle),
				"centroid_x": x,
				"centroid_y": y,
				"mean_angle": angle,
				"sample_ids": {sample_idx},
				"kind_weights": {minutia.kind: combined_weight},
			})
	
	# Filter by consensus and create final minutiae
	fused_minutiae: List[Minutia] = []
	for cluster in clusters:
		consensus_ratio = len(cluster["sample_ids"]) / float(n_samples)
		if consensus_ratio < min_consensus_adjusted:
			continue
		
		x = cluster["sum_x"] / cluster["sum_weights"]
		y = cluster["sum_y"] / cluster["sum_weights"]
		angle = math.atan2(cluster["sum_sin"], cluster["sum_cos"])
		
		# Determine kind by majority vote
		kind_weights = cluster["kind_weights"]
		kind = max(kind_weights.items(), key=lambda item: item[1])[0] if kind_weights else "ending"
		
		fused_minutiae.append(Minutia(
			x=float(x),
			y=float(y),
			angle=float(angle),
			kind=kind,
			quality=float(np.clip(consensus_ratio, 0.0, 1.0)),
		))
	
	print(f"[fusion] Clustered {sum(len(s) for s in aligned_minutiae_sets)} minutiae into {len(fused_minutiae)} fused minutiae")
	
	return fused_minutiae


# ---------------------------------------------------------------------------
# Template Fusion Pipeline


class TemplateFusionPipeline:
	"""Pipeline for fusing multiple fingerprint samples into a single template."""
	
	def __init__(
		self,
		base_pipeline: FingerprintPipeline,
		fusion_settings: FusionSettings,
	) -> None:
		"""Initialize fusion pipeline.
		
		Args:
			base_pipeline: Base fingerprint processing pipeline
			fusion_settings: Fusion configuration settings
		"""
		self.base_pipeline = base_pipeline
		self.fusion_settings = fusion_settings
	
	def process_user(
		self,
		image_paths: Sequence[Path],
		identifier: str,
	) -> Optional[FingerprintTemplate]:
		"""Process multiple images of same user into fused template.
		
		Args:
			image_paths: List of fingerprint image paths for same user
			identifier: User identifier
			
		Returns:
			Fused template or None if fusion fails
		"""
		if not image_paths:
			return None
		
		# Step 1: Process all images
		print(f"[fusion] Processing {len(image_paths)} samples for user {identifier}")
		templates: List[FingerprintTemplate] = []
		quality_assessments: List[float] = []
		
		for path in image_paths:
			template = self.base_pipeline.process(path, identifier)
			
			# Load image for quality assessment
			image = load_grayscale_image(path)
			normalised = normalise_image(image)
			mask = block_variance_segmentation(normalised)
			diffused = coherence_diffusion(normalised, mask)
			orientation, frequency = estimate_orientation_and_frequency(diffused, mask)
			enhanced = apply_log_gabor_enhancement(diffused, mask, orientation, frequency)
			binary, skeleton = binarise_and_thin(enhanced, mask)
			
			# Assess quality
			quality = assess_image_quality(enhanced, mask, skeleton, template.minutiae or [])
			quality_assessments.append(quality)
			
			if quality >= FUSION_MIN_QUALITY:
				templates.append(template)
				print(f"[fusion]   {path.name}: quality={quality:.3f} OK")
			else:
				print(f"[fusion]   {path.name}: quality={quality:.3f} (skipped, below {FUSION_MIN_QUALITY})")
		
		if not templates:
			print(f"[fusion] No valid templates for {identifier}")
			return None
		
		if len(templates) == 1:
			print(f"[fusion] Only one valid template, returning as-is")
			return templates[0]
		
		# Step 2: Select reference (highest quality)
		valid_qualities = [q for q, t in zip(quality_assessments, templates) if q >= FUSION_MIN_QUALITY]
		reference_idx = int(np.argmax(valid_qualities))
		reference = templates[reference_idx]
		print(f"[fusion] Selected reference: sample {reference_idx + 1} (quality={valid_qualities[reference_idx]:.3f})")
		
		# Step 3: Align minutiae sets to reference
		aligned_minutiae_sets: List[List[Minutia]] = []
		alignment_confidences: List[float] = []
		
		image_shape = (300, 300)  # Default shape, will be overridden
		if reference.minutiae:
			for idx, template in enumerate(templates):
				if template.minutiae is None:
					continue
				
				if idx == reference_idx:
					aligned_minutiae_sets.append(list(template.minutiae))
					alignment_confidences.append(1.0)
				else:
					aligned, confidence = align_minutiae_sets(
						reference.minutiae,
						template.minutiae,
						image_shape,
					)
					aligned_minutiae_sets.append(aligned)
					alignment_confidences.append(confidence)
					print(f"[fusion]   Aligned sample {idx + 1}: confidence={confidence:.3f}")
		
		# Step 4: Fuse based on mode
		fused_vector = None
		fused_minutiae: List[Minutia] = []
		
		if self.fusion_settings.mode in ("features", "full"):
			fused_vector = fuse_feature_vectors(templates)
			print(f"[fusion] Feature vectors fused: {len(templates)} templates")
		else:
			fused_vector = reference.raw_features
		
		if self.fusion_settings.mode in ("minutiae", "full"):
			if aligned_minutiae_sets:
				fused_minutiae = fuse_minutiae_advanced(aligned_minutiae_sets, valid_qualities)
				print(f"[fusion] Minutiae fused: {len(fused_minutiae)} from {len(aligned_minutiae_sets)} sets")
			else:
				fused_minutiae = list(reference.minutiae or [])
		else:
			fused_minutiae = list(reference.minutiae or [])
		
		if fused_vector is None:
			return None
		
		# Step 5: Compute final quality
		qualities_array = np.array(valid_qualities, dtype=np.float32)
		avg_quality = float(qualities_array.mean())
		consensus_score = float(np.mean([m.quality for m in fused_minutiae])) if fused_minutiae else 0.0
		
		# Step 6: Encode protected template
		protected = self.base_pipeline.hasher.encode(fused_vector)
		
		print(f"[fusion] Final template: quality={avg_quality:.3f}, consensus={consensus_score:.3f}")
		
		return FingerprintTemplate(
			identifier=identifier,
			image_path=Path(f"{identifier}__fused"),
			protected=protected,
			bit_length=self.base_pipeline.hasher.bit_length,
			quality=avg_quality,
			raw_features=fused_vector,
			minutiae=fused_minutiae,
			fused=True,
			source_count=len(templates),
			consensus_score=consensus_score,
		)


def fuse_protected_templates(
	templates: Sequence[FingerprintTemplate],
	hasher: CancelableHasher,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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


def fuse_minutiae_consensus(
	templates: Sequence[FingerprintTemplate],
	settings: FusionSettings,
) -> List[Minutia]:
	if not templates:
		return []
	clusters: List[dict] = []
	for sample_idx, template in enumerate(templates):
		sample_quality = max(float(template.quality), 1e-6)
		for minutia in template.minutiae or []:
			x = float(minutia.x)
			y = float(minutia.y)
			angle = float(minutia.angle)
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
				if angle_diff > settings.angle_rad:
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
				clusters.append(
					{
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
					}
				)
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
		if kind_weights:
			kind = max(kind_weights.items(), key=lambda item: item[1])[0]
		else:
			kind = "ending"
		fused_minutiae.append(
			Minutia(
				x=float(x),
				y=float(y),
				angle=float(angle),
				kind=kind,
				quality=float(np.clip(consensus_ratio, 0.0, 1.0)),
			)
		)
	return fused_minutiae


def create_fused_template(
	identifier: str,
	templates: Sequence[FingerprintTemplate],
	hasher: CancelableHasher,
	settings: FusionSettings,
) -> Optional[FingerprintTemplate]:
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
	avg_quality = float(
		np.clip(np.dot(template_weights, np.array([t.quality for t in templates], dtype=np.float32)), 0.0, 1.0)
	)
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
	"""Estimate a 0-1 quality score for the fingerprint sample."""
	if mask.size == 0:
		return 0.0
	mask_bool = mask.astype(bool)
	foreground_ratio = float(mask_bool.mean()) if mask_bool.size else 0.0
	skeleton_ratio = float(np.clip(skeleton.astype(np.float32).mean(), 0.0, 1.0)) if skeleton.size else 0.0
	minutiae_ratio = 0.0
	if MAX_MINUTIAE_ENCODER > 0:
		minutiae_ratio = min(len(minutiae), MAX_MINUTIAE_ENCODER) / float(MAX_MINUTIAE_ENCODER)
	orientation_array = np.asarray(orientation, dtype=np.float32)
	orientation_valid = 0.0
	if orientation_array.size:
		valid_mask = np.isfinite(orientation_array) & (np.abs(orientation_array) > 1e-3)
		orientation_valid = float(np.count_nonzero(valid_mask)) / float(orientation_array.size)
	frequency_array = np.asarray(frequency, dtype=np.float32)
	freq_valid = 0.0
	if frequency_array.size:
		valid_freq = np.isfinite(frequency_array) & (frequency_array > 0.0)
		freq_valid = float(np.count_nonzero(valid_freq)) / float(frequency_array.size)
	contrast = 0.0
	diffused_contrast = 0.0
	if mask_bool.any():
		roi = enhanced[mask_bool]
		roi_std = float(np.std(roi)) if roi.size else 0.0
		contrast = float(np.clip(roi_std / 64.0, 0.0, 1.0))
		diff_roi = diffused[mask_bool]
		diff_std = float(np.std(diff_roi)) if diff_roi.size else 0.0
		diffused_contrast = float(np.clip(diff_std / 64.0, 0.0, 1.0))
	level3_density = 0.0
	if level3:
		level3_density = min(len(level3) / 25.0, 1.0)
	components = (
		(0.23, minutiae_ratio),
		(0.20, foreground_ratio),
		(0.14, skeleton_ratio),
		(0.14, orientation_valid),
		(0.10, freq_valid),
		(0.09, contrast),
		(0.05, diffused_contrast),
	)
	score = sum(weight * value for weight, value in components)
	score += 0.05 * level3_density
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
) -> np.ndarray:
	if mask.ndim != 2:
		raise ValueError("Mask must be a 2-D array.")
	height, width = mask.shape
	width = max(width, 1)
	height = max(height, 1)
	mask_bool = mask.astype(bool)
	features: List[float] = []

	sorted_minutiae = sorted(minutiae, key=lambda m: m.quality, reverse=True)
	for idx in range(MAX_MINUTIAE_ENCODER):
		if idx < len(sorted_minutiae):
			minutia = sorted_minutiae[idx]
			norm_quality = float(np.clip(minutia.quality, 0.0, 255.0)) / 255.0
			features.extend(
				[
					float(minutia.x) / float(width),
					float(minutia.y) / float(height),
					math.cos(minutia.angle),
					math.sin(minutia.angle),
					norm_quality,
				]
			)
		else:
			features.extend([0.0] * MINUTIA_FEATURE_SIZE)

	minutiae_count = len(sorted_minutiae)
	features.append(minutiae_count / float(MAX_MINUTIAE_ENCODER or 1))
	features.append(float(mask.mean()))
	features.append(float(skeleton.mean()))

	if sorted_minutiae:
		qualities = np.array([m.quality for m in sorted_minutiae[:MAX_MINUTIAE_ENCODER]], dtype=np.float32) / 255.0
		xs = np.array([m.x for m in sorted_minutiae[:MAX_MINUTIAE_ENCODER]], dtype=np.float32) / float(width)
		ys = np.array([m.y for m in sorted_minutiae[:MAX_MINUTIAE_ENCODER]], dtype=np.float32) / float(height)
		features.append(float(qualities.mean()))
		features.append(float(qualities.std()))
		features.append(float(xs.mean()))
		features.append(float(xs.std()))
		features.append(float(ys.mean()))
		features.append(float(ys.std()))
		features.append(float(xs.max() - xs.min()))
		features.append(float(ys.max() - ys.min()))
	else:
		features.extend([0.0] * 8)

	orientation_values = orientation.astype(np.float32).flatten()
	orientation_values = orientation_values[np.isfinite(orientation_values)]
	if orientation_values.size:
		hist_orientation, _ = np.histogram(
			orientation_values,
			bins=ORI_HIST_BINS,
			range=(-math.pi, math.pi),
			density=True,
		)
		mean_cos = float(np.mean(np.cos(orientation_values)))
		mean_sin = float(np.mean(np.sin(orientation_values)))
		orientation_coherence = math.sqrt(mean_cos ** 2 + mean_sin ** 2)
		orientation_std = float(np.std(orientation_values))
		orientation_coverage = float(np.count_nonzero(np.abs(orientation_values)) / orientation_values.size)
	else:
		hist_orientation = np.zeros(ORI_HIST_BINS, dtype=np.float32)
		mean_cos = 0.0
		mean_sin = 0.0
		orientation_coherence = 0.0
		orientation_std = 0.0
		orientation_coverage = 0.0
	features.extend(hist_orientation.astype(np.float32))
	features.extend([mean_cos, mean_sin, orientation_coherence, orientation_std, orientation_coverage])

	frequency_values = frequency.astype(np.float32).flatten()
	valid_freq = frequency_values[(frequency_values > 0.0) & np.isfinite(frequency_values)]
	if valid_freq.size:
		upper = float(max(valid_freq.max(), 0.1))
		hist_frequency, _ = np.histogram(
			valid_freq,
			bins=FREQ_HIST_BINS,
			range=(0.0, upper),
			density=True,
		)
		freq_mean = float(valid_freq.mean())
		freq_std = float(valid_freq.std())
		freq_coverage = float(np.count_nonzero(valid_freq) / valid_freq.size)
	else:
		hist_frequency = np.zeros(FREQ_HIST_BINS, dtype=np.float32)
		freq_mean = 0.0
		freq_std = 0.0
		freq_coverage = 0.0
	features.extend(hist_frequency.astype(np.float32))
	features.extend([freq_mean, freq_std, freq_coverage])

	if level3:
		radii = np.array([pore.radius for pore in level3], dtype=np.float32)
		strengths = np.array([pore.strength for pore in level3], dtype=np.float32)
		features.extend([
			float(radii.mean()) if radii.size else 0.0,
			float(radii.std()) if radii.size else 0.0,
			float(strengths.mean()) if strengths.size else 0.0,
			float(len(level3)),
		])
	else:
		features.extend([0.0, 0.0, 0.0, 0.0])

	foreground = enhanced[mask_bool]
	diffused_foreground = diffused[mask_bool]
	if foreground.size:
		features.append(float(foreground.mean()) / 255.0)
		features.append(float(foreground.std()) / 255.0)
	else:
		features.extend([0.0, 0.0])
	if diffused_foreground.size:
		features.append(float(diffused_foreground.mean()) / 255.0)
		features.append(float(diffused_foreground.std()) / 255.0)
	else:
		features.extend([0.0, 0.0])

	for kernel in GABOR_KERNELS:
		response = cv2.filter2D(enhanced, cv2.CV_32F, kernel)
		masked = np.abs(response)[mask_bool]
		if masked.size:
			features.append(float(masked.mean()))
			features.append(float(masked.std()))
		else:
			features.extend([0.0, 0.0])

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
# Template persistence helpers


def template_output_path(output_dir: Path, image_path: Path) -> Path:
	return output_dir / f"{image_path.stem}{TEMPLATE_EXTENSION}"


def save_template(template: FingerprintTemplate, output_dir: Path, *, overwrite: bool = False) -> Path:
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
	if not directory.exists() or not directory.is_dir():
		raise FileNotFoundError(f"Template directory not found: {directory}")
	templates: List[FingerprintTemplate] = []
	for file_path in sorted(directory.glob(f"*{TEMPLATE_EXTENSION}")):
		try:
			with file_path.open("rb") as handle:
				loaded = pickle.load(handle)
		except Exception as exc:  # noqa: BLE001
			print(f"[warning] Failed to load template {file_path.name}: {exc}")
			continue
		if isinstance(loaded, FingerprintTemplate):
			templates.append(loaded)
		else:
			print(f"[warning] Ignoring {file_path.name}: incompatible template format")
	if not templates:
		return templates
	grouped: dict[str, List[FingerprintTemplate]] = {}
	for template in templates:
		grouped.setdefault(template.identifier, []).append(template)
	resolved: List[FingerprintTemplate] = []
	for identifier, group in grouped.items():
		fused_templates = [t for t in group if t.fused]
		non_fused = [t for t in group if not t.fused]
		if prefer_fused and fused_templates:
			resolved.extend(fused_templates)
			if include_raw_when_fused and non_fused:
				resolved.extend(non_fused)
		elif not prefer_fused and non_fused:
			resolved.extend(non_fused)
		else:
			resolved.extend(group)
	return resolved


def build_template_cache(pipeline: FingerprintPipeline,
					 raw_dir: Path,
					 output_dir: Path,
					 *,
					 overwrite: bool = False,
					 quality_threshold: float = 0.0,
			 fusion: Optional[FusionSettings] = None) -> Tuple[int, int, int, int, int]:
	image_paths = enumerate_database(raw_dir)
	total = len(image_paths)
	processed = 0
	skipped_existing = 0
	skipped_low_quality = 0
	fused_created = 0
	
	# Group images by identity
	paths_by_identity: dict[str, List[Path]] = {}
	for path in image_paths:
		identifier = infer_identity_from_name(path)
		paths_by_identity.setdefault(identifier, []).append(path)
	
	fusion_settings = fusion or FusionSettings()
	
	# Check if we should use advanced fusion pipeline
	use_advanced_fusion = (
		fusion_settings.enabled and 
		fusion_settings.mode in ("minutiae", "features", "full")
	)
	
	if use_advanced_fusion:
		fusion_pipeline = TemplateFusionPipeline(pipeline, fusion_settings)
	
	for identifier, identity_paths in paths_by_identity.items():
		# Use advanced fusion if enabled
		if use_advanced_fusion and len(identity_paths) > 1:
			fused_template = fusion_pipeline.process_user(identity_paths, identifier)
			if fused_template is not None and fused_template.quality >= quality_threshold:
				save_template(fused_template, output_dir, overwrite=True)
				fused_created += 1
				processed += len(identity_paths)
			else:
				if fused_template is not None:
					print(f"[warning] Fused template for {identifier} below quality threshold")
				skipped_low_quality += len(identity_paths)
			continue
		
		# Original per-image processing
		identity_templates: List[FingerprintTemplate] = []
		for path in identity_paths:
			target = template_output_path(output_dir, path)
			template: Optional[FingerprintTemplate] = None
			reuse_existing = False
			
			if target.exists() and not overwrite:
				try:
					with target.open("rb") as handle:
						loaded = pickle.load(handle)
				except Exception as exc:  # noqa: BLE001
					print(f"[warning] Failed to reuse template {target.name}: {exc}")
				else:
					if isinstance(loaded, FingerprintTemplate):
						if loaded.quality < quality_threshold:
							print(
								f"[warning] Removing cached template {target.name}: quality {loaded.quality:.3f} "
								f"below threshold {quality_threshold:.3f}."
							)
							try:
								target.unlink()
							except OSError as cleanup_exc:  # noqa: PERF203
								print(
									f"[warning] Failed to remove outdated template {target.name}: {cleanup_exc}"
								)
						else:
							template = loaded
							reuse_existing = True
					else:
						print(f"[warning] Ignoring cached file {target.name}: incompatible format")
			
			if template is None:
				template = pipeline.process(path, identifier)
				if template.quality < quality_threshold:
					print(
						f"[warning] Skipping {path.name}: quality {template.quality:.3f} below threshold "
						f"{quality_threshold:.3f}."
					)
					if target.exists():
						try:
							target.unlink()
						except OSError as exc:  # noqa: PERF203
							print(f"[warning] Failed to remove outdated template {target.name}: {exc}")
					skipped_low_quality += 1
					continue
				save_template(template, output_dir, overwrite=True)
				processed += 1
			else:
				if reuse_existing:
					skipped_existing += 1
			
			identity_templates.append(template)
		
		# Simple fusion for backward compatibility
		if not use_advanced_fusion and fusion_settings.enabled and len(identity_templates) > 1:
			fused_template = create_fused_template(identifier, identity_templates, pipeline.hasher, fusion_settings)
			if fused_template is not None:
				save_template(fused_template, output_dir, overwrite=True)
				fused_created += 1
	
	return processed, skipped_existing, skipped_low_quality, total, fused_created


# ---------------------------------------------------------------------------
# Pipeline orchestrator


class FingerprintPipeline:
	def __init__(self, include_level3: bool = False, hasher: Optional[CancelableHasher] = None) -> None:
		self.include_level3 = include_level3
		if hasher is None:
			raise ValueError("CancelableHasher instance must be provided for protected templates.")
		self.hasher = hasher

	def process(self, image_path: Path, identifier: str) -> FingerprintTemplate:
		"""Process fingerprint image - REVERTED to original pipeline."""
		image = load_grayscale_image(image_path)
		
		normalised = normalise_image(image)
		mask = block_variance_segmentation(normalised)
		diffused = coherence_diffusion(normalised, mask)

		orientation, frequency = estimate_orientation_and_frequency(diffused, mask)
		enhanced = apply_log_gabor_enhancement(diffused, mask, orientation, frequency)

		binary, skeleton = binarise_and_thin(enhanced, mask)

		minutiae = extract_minutiae(skeleton, mask, orientation)
		minutiae = validate_minutiae(minutiae, skeleton, mask)

		level3_features: List[Pore] = []
		if self.include_level3:
			level3_features = detect_pores(enhanced, mask)

		quality_score = evaluate_quality(
			minutiae,
			mask,
			orientation,
			frequency,
			skeleton,
			enhanced,
			diffused,
			level3=level3_features if self.include_level3 else None,
		)

		feature_vector = build_feature_vector(
			minutiae,
			mask,
			orientation,
			frequency,
			skeleton,
			enhanced,
			diffused,
			level3=level3_features if self.include_level3 else None,
		)
		protected = self.hasher.encode(feature_vector)

		return FingerprintTemplate(
			identifier=identifier,
			image_path=image_path,
			protected=protected,
			bit_length=self.hasher.bit_length,
			quality=quality_score,
			raw_features=feature_vector.astype(np.float32, copy=True),
			minutiae=list(minutiae),
			consensus_score=1.0,
		)


# ---------------------------------------------------------------------------
# Geometric Minutiae Matching (Two-Stage Verification)


def compute_geometric_minutiae_score(
	probe_minutiae: Sequence[Minutia],
	candidate_minutiae: Sequence[Minutia],
	distance_threshold: float = MATCH_DISTANCE_THRESHOLD,
	angle_threshold: float = MATCH_ANGLE_THRESHOLD_RAD,
) -> float:
	"""Compute geometric similarity score between two minutiae sets.
	
	This function performs direct minutiae-to-minutiae matching using
	spatial and angular correspondence after RANSAC alignment.
	
	Args:
		probe_minutiae: Probe minutiae list
		candidate_minutiae: Candidate minutiae list
		distance_threshold: Max distance for minutiae correspondence (pixels)
		angle_threshold: Max angle difference for correspondence (radians)
	
	Returns:
		Geometric similarity score in [0, 1] range
	"""
	if not probe_minutiae or not candidate_minutiae:
		return 0.0
	
	n_probe = len(probe_minutiae)
	n_candidate = len(candidate_minutiae)
	
	# Extract positions and angles
	probe_points = np.array([[m.x, m.y] for m in probe_minutiae], dtype=np.float32)
	cand_points = np.array([[m.x, m.y] for m in candidate_minutiae], dtype=np.float32)
	
	# Step 1: Find transformation using RANSAC
	if n_probe < 3 or n_candidate < 3:
		# Not enough points for RANSAC, use simple nearest neighbor
		tree = KDTree(cand_points)
		distances, _ = tree.query(probe_points, k=1)
		matches = np.sum(distances < distance_threshold)
		return float(matches) / float(max(n_probe, n_candidate))
	
	# RANSAC to find correspondences
	best_inliers = 0
	best_transform = None
	max_iterations = min(100, n_probe * 2)
	
	for _ in range(max_iterations):
		# Sample 3 random probe minutiae
		if n_probe < 3:
			break
		sample_indices = np.random.choice(n_probe, size=3, replace=False)
		probe_sample = probe_points[sample_indices]
		
		# Find nearest candidate minutiae for each sample
		tree = KDTree(cand_points)
		distances, cand_indices = tree.query(probe_sample, k=1)
		
		# Skip if correspondences are too far
		if np.any(distances > distance_threshold * 2):
			continue
		
		cand_sample = cand_points[cand_indices]
		
		# Estimate rigid transformation (rotation + translation)
		try:
			# Compute centroids
			probe_centroid = probe_sample.mean(axis=0)
			cand_centroid = cand_sample.mean(axis=0)
			
			# Center points
			probe_centered = probe_sample - probe_centroid
			cand_centered = cand_sample - cand_centroid
			
			# Compute rotation using SVD
			H = probe_centered.T @ cand_centered
			U, _, Vt = np.linalg.svd(H)
			R = Vt.T @ U.T
			
			# Ensure proper rotation (det(R) = 1)
			if np.linalg.det(R) < 0:
				Vt[-1, :] *= -1
				R = Vt.T @ U.T
			
			# Compute translation
			t = cand_centroid - R @ probe_centroid
			
			# Transform all probe points
			transformed = (R @ probe_points.T).T + t
			
			# Count inliers (both spatial and angular)
			tree_full = KDTree(cand_points)
			distances_full, indices_full = tree_full.query(transformed, k=1)
			
			# Check spatial distance
			spatial_inliers = distances_full < distance_threshold
			
			# Check angular consistency for spatial inliers
			rotation_angle = math.atan2(R[1, 0], R[0, 0])
			angular_inliers = np.zeros(n_probe, dtype=bool)
			
			for i, (is_spatial, cand_idx) in enumerate(zip(spatial_inliers, indices_full)):
				if not is_spatial:
					continue
				
				# Compare angles
				probe_angle = probe_minutiae[i].angle
				cand_angle = candidate_minutiae[cand_idx].angle
				
				# Rotate probe angle
				rotated_angle = (probe_angle + rotation_angle) % (2.0 * math.pi)
				
				# Compute angle difference
				angle_diff = abs(rotated_angle - cand_angle)
				angle_diff = min(angle_diff, 2.0 * math.pi - angle_diff)
				
				angular_inliers[i] = angle_diff < angle_threshold
			
			# Combined inliers (spatial AND angular)
			combined_inliers = spatial_inliers & angular_inliers
			inlier_count = np.sum(combined_inliers)
			
			if inlier_count > best_inliers:
				best_inliers = inlier_count
				best_transform = (R, t, rotation_angle)
		
		except (np.linalg.LinAlgError, ValueError):
			continue
	
	if best_transform is None or best_inliers < 3:
		# Fallback: simple nearest neighbor without transformation
		tree = KDTree(cand_points)
		distances, _ = tree.query(probe_points, k=1)
		matches = np.sum(distances < distance_threshold)
		score = float(matches) / float(max(n_probe, n_candidate))
		return score * 0.5  # Penalize because no good alignment found
	
	# Step 2: Compute final score based on inliers
	# Normalize by the larger set to penalize size mismatch
	max_minutiae = max(n_probe, n_candidate)
	geometric_score = float(best_inliers) / float(max_minutiae)
	
	# Bonus for high inlier ratio in both sets
	probe_ratio = float(best_inliers) / float(n_probe)
	candidate_ratio = float(best_inliers) / float(n_candidate)
	
	# Weighted combination
	final_score = 0.6 * geometric_score + 0.2 * probe_ratio + 0.2 * candidate_ratio
	
	return float(np.clip(final_score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Matching engine


class FingerprintMatcher:
	def __init__(self, templates: Sequence[FingerprintTemplate], hasher: CancelableHasher) -> None:
		self.templates = list(templates)
		self.hasher = hasher
		expected_bits = self.hasher.bit_length
		for template in self.templates:
			if template.bit_length != expected_bits:
				raise ValueError(
					"Template bit-length mismatch. Rebuild the cache with the current projection key."
				)
	
	def adaptive_threshold(self, probe_quality: float, candidate_quality: float) -> float:
		"""Compute adaptive matching threshold based on probe and candidate quality.
		
		TUNED: Less aggressive adjustment than before.
		
		Args:
			probe_quality: Quality score of probe template [0, 1]
			candidate_quality: Quality score of candidate template [0, 1]
		
		Returns:
			Adjusted matching threshold
		"""
		# Both high quality: use standard threshold (0.78)
		if probe_quality > 0.7 and candidate_quality > 0.7:
			return MATCH_MIN_SCORE
		
		# Both medium quality: relax slightly (was -0.05, now -0.03)
		if probe_quality >= 0.5 and candidate_quality >= 0.5:
			adjusted = MATCH_MIN_SCORE - 0.03  # 0.75
			return adjusted
		
		# At least one low quality: relax more (was -0.08, now -0.05)
		adjusted = MATCH_MIN_SCORE - 0.05  # 0.73
		return adjusted

	def identify(self, probe: FingerprintTemplate, top_k: int = 5, use_geometric_reranking: bool = True, min_probe_quality: float = 0.35) -> List[Tuple[str, float]]:
		"""Two-stage identification: hash-based matching + geometric verification.
		
		Args:
			probe: Probe fingerprint template
			top_k: Number of top matches to return
			use_geometric_reranking: If True, re-rank top candidates using geometric minutiae matching
			min_probe_quality: Minimum probe quality to accept (default 0.35). Lower quality probes
			                   may produce unreliable results and should trigger recapture.
		
		Returns:
			List of (identifier, score) tuples sorted by reranked order
			Note: When geometric reranking is applied, top-N scores are combined (60% hash + 40% geometric)
			      Remaining candidates use normalized scores (60% of hash score)
		
		Raises:
			ValueError: If probe quality is below min_probe_quality threshold
		"""
		# Quality check: reject low-quality probes
		if probe.quality < min_probe_quality:
			raise ValueError(
				f"Probe quality {probe.quality:.3f} is below minimum threshold {min_probe_quality:.3f}. "
				f"Please recapture the fingerprint with better quality."
			)
		
		# Stage 1: Hash-based matching (fast, all candidates)
		hash_scores: List[Tuple[str, float, FingerprintTemplate]] = []
		for candidate in self.templates:
			if candidate.image_path == probe.image_path:
				continue
			hash_similarity = self.hasher.similarity(probe.protected, candidate.protected)
			hash_scores.append((candidate.image_path.name, hash_similarity, candidate))
		
		hash_scores.sort(key=lambda item: item[1], reverse=True)
		
		# If geometric reranking disabled or too few candidates, return hash scores
		if not use_geometric_reranking or len(hash_scores) < 2:
			return [(name, score) for name, score, _ in hash_scores[:top_k]]
		
		# Stage 2: Geometric verification for top candidates (slow, but accurate)
		# Start with top-3, but expand to include tied candidates
		top_candidates = hash_scores[:min(3, len(hash_scores))]
		
		# Detect ties: if candidates beyond position 3 have very similar scores, include them
		if len(hash_scores) > 3:
			third_score = hash_scores[2][1]
			tie_threshold = 0.0005  # Very tight threshold for detecting ties
			
			# Check positions 4-6 for ties with position 3
			for i in range(3, min(len(hash_scores), 6)):
				score_diff = abs(hash_scores[i][1] - third_score)
				if score_diff < tie_threshold:
					top_candidates.append(hash_scores[i])
					print(f"[geometric] Including tied candidate at position {i+1}: {hash_scores[i][0]} (diff={score_diff:.6f})")
				else:
					break  # Stop at first non-tied candidate
		
		if len(top_candidates) >= 2:
			score_diff = top_candidates[0][1] - top_candidates[1][1]
			
			# Only apply geometric reranking if scores are ambiguous
			if score_diff < 0.02:
				print(f"[geometric] Hash scores close (diff={score_diff:.4f}), applying geometric verification to top-{len(top_candidates)}...")
				
				# Compute geometric scores for reranking
				reranking_data: List[Tuple[str, float, float, float]] = []
				for name, hash_score, candidate in top_candidates:
					# Compute geometric similarity
					geometric_score = compute_geometric_minutiae_score(
						probe.minutiae,
						candidate.minutiae,
						distance_threshold=MATCH_DISTANCE_THRESHOLD,
						angle_threshold=MATCH_ANGLE_THRESHOLD_RAD,
					)
					
					# Combined score: 60% hash + 40% geometric
					combined_score = 0.6 * hash_score + 0.4 * geometric_score
					
					reranking_data.append((name, hash_score, geometric_score, combined_score))
					print(f"[geometric]   {name}: hash={hash_score:.4f}, geom={geometric_score:.4f}, combined={combined_score:.4f}")
				
				# Re-rank by combined score
				reranking_data.sort(key=lambda item: item[3], reverse=True)
				
				# Build final result: reranked top-N with COMBINED scores + remaining with normalized scores
				final_scores: List[Tuple[str, float]] = []
				reranked_names = set()
				
				# Add reranked candidates with COMBINED scores
				for name, hash_score, geom_score, combined in reranking_data:
					final_scores.append((name, combined))
					reranked_names.add(name)
					print(f"[geometric] Reranked: {name} (combined={combined:.4f})")
				
				# Add remaining candidates with normalized scores (hash only = 60% hash + 40% * 0)
				# This keeps scores in same scale as reranked candidates
				for name, hash_score, _ in hash_scores:
					if name not in reranked_names:
						# Normalize to same scale: 0.6 * hash_score (geometric component = 0)
						normalized_score = 0.6 * hash_score
						final_scores.append((name, normalized_score))
				
				return final_scores[:top_k]
		
		# No reranking needed (scores not close), return hash scores
		return [(name, score) for name, score, _ in hash_scores[:top_k]]

	def verify(self, probe: FingerprintTemplate, claimed_id: str) -> Tuple[bool, float]:
		candidates = [t for t in self.templates if t.identifier == claimed_id]
		if not candidates:
			raise ValueError(f"No templates available for claimed identity '{claimed_id}'.")
		best_score = 0.0
		for candidate in candidates:
			similarity = self.hasher.similarity(probe.protected, candidate.protected)
			best_score = max(best_score, similarity)
		return best_score >= MATCH_MIN_SCORE, best_score
		
		return best_score >= threshold, best_score


# ---------------------------------------------------------------------------
# Dataset helpers


def enumerate_database(db_path: Path) -> List[Path]:
	if not db_path.exists() or not db_path.is_dir():
		raise FileNotFoundError(f"Fingerprint database not found: {db_path}")
	files = [p for p in db_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
	if not files:
		raise FileNotFoundError(f"No fingerprint images found inside {db_path}")
	files.sort()
	return files


def infer_identity_from_name(path: Path) -> str:
	stem = path.stem
	if "_" in stem:
		return stem.split("_")[0]
	return stem


def build_gallery(
	pipeline: FingerprintPipeline,
	image_paths: Sequence[Path],
	*,
	fusion: Optional[FusionSettings] = None,
) -> List[FingerprintTemplate]:
	templates: List[FingerprintTemplate] = []
	for path in image_paths:
		identifier = infer_identity_from_name(path)
		template = pipeline.process(path, identifier)
		templates.append(template)
	if fusion is not None and fusion.enabled:
		return fuse_identity_templates(templates, pipeline.hasher, fusion)
	return templates


# ---------------------------------------------------------------------------
# Command line interface


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Hybrid fingerprint matcher for identification/authentication.")
	parser.add_argument("--probe", type=Path, default=None, help="Fingerprint image to analyse.")
	parser.add_argument("--mode", choices=["identify", "verify"], default="identify", help="Operation mode.")
	parser.add_argument("--claim", type=str, default=None, help="Claimed identity for verification mode.")
	parser.add_argument("--top", type=int, default=5, help="Number of top matches to return in identification mode.")
	parser.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATE_PATH, help="Directory containing cached fingerprint templates.")
	parser.add_argument("--build-from", type=Path, default=None, help="Process raw fingerprints from this directory into template files.")
	parser.add_argument("--overwrite-templates", action="store_true", help="Rebuild templates even if cache files already exist.")
	parser.add_argument("--use-level3", action="store_true", help="Enable Level-3 feature fusion (requires high-resolution images).")
	parser.add_argument(
		"--projection-key",
		type=str,
		default="default",
		help="Cancelable projection key. Use the same value for building and matching to keep templates valid.",
	)
	parser.add_argument(
		"--projection-dim",
		type=int,
		default=DEFAULT_PROJECTION_DIM,
		help="Number of bits per hash projection (default: %(default)s).",
	)
	parser.add_argument(
		"--hash-count",
		type=int,
		default=DEFAULT_HASH_COUNT,
		help="Number of independent projection hashes to fuse (default: %(default)s).",
	)
	parser.add_argument(
		"--quality-threshold",
		type=float,
		default=DEFAULT_QUALITY_THRESHOLD,
		help="Minimum template quality (0-1) required for saving and matching (default: %(default)s).",
	)
	parser.add_argument("--no-fusion", action="store_true", help="Disable template fusion.")
	parser.add_argument(
		"--fusion-mode",
		type=str,
		choices=["none", "minutiae", "features", "full"],
		default="full",
		help="Fusion mode: none (disabled), minutiae (only minutiae), features (only vectors), full (both) (default: %(default)s).",
	)
	parser.add_argument(
		"--fusion-distance",
		type=float,
		default=DEFAULT_FUSION_DISTANCE,
		help="Maximum spatial distance (pixels) when clustering minutiae for consensus (default: %(default)s).",
	)
	parser.add_argument(
		"--fusion-angle",
		type=float,
		default=DEFAULT_FUSION_ANGLE_DEG,
		help="Maximum angular difference (degrees) for minutiae clustering (default: %(default)s).",
	)
	parser.add_argument(
		"--fusion-min-consensus",
		type=float,
		default=DEFAULT_FUSION_MIN_CONSENSUS,
		help="Minimum fraction of samples that must support a minutia to keep it (default: %(default)s).",
	)
	parser.add_argument(
		"--fusion-keep-raw",
		action="store_true",
		help="Keep individual templates alongside fused ones in memory operations.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.projection_dim <= 0:
		raise SystemExit("--projection-dim must be positive.")
	if args.hash_count <= 0:
		raise SystemExit("--hash-count must be positive.")
	if not (0.0 <= args.quality_threshold <= 1.0):
		raise SystemExit("--quality-threshold must be between 0 and 1.")
	if args.fusion_distance <= 0.0:
		raise SystemExit("--fusion-distance must be positive.")
	if args.fusion_angle <= 0.0 or args.fusion_angle > 180.0:
		raise SystemExit("--fusion-angle must be in the range (0, 180].")
	if not (0.0 < args.fusion_min_consensus <= 1.0):
		raise SystemExit("--fusion-min-consensus must be within (0, 1].")
	
	# Determine fusion mode
	fusion_mode = "none" if args.no_fusion else args.fusion_mode
	
	fusion_settings = FusionSettings(
		enabled=(fusion_mode != "none"),
		distance=args.fusion_distance,
		angle_deg=args.fusion_angle,
		min_consensus=args.fusion_min_consensus,
		keep_raw=args.fusion_keep_raw,
		mode=fusion_mode,
	)
	hasher = CancelableHasher(
		feature_dim=FEATURE_VECTOR_DIM,
		projection_dim=args.projection_dim,
		key=args.projection_key,
		hash_count=args.hash_count,
	)
	pipeline = FingerprintPipeline(include_level3=args.use_level3, hasher=hasher)
	templates_dir = args.templates

	if args.build_from is not None:
		try:
			processed, skipped_existing, skipped_low_quality, total, fused_created = build_template_cache(
				pipeline,
				raw_dir=args.build_from,
				output_dir=templates_dir,
				overwrite=args.overwrite_templates,
				quality_threshold=args.quality_threshold,
				fusion=fusion_settings,
			)
		except FileNotFoundError as exc:
			raise SystemExit(str(exc)) from exc
		summary_parts = [
			f"processed {processed} of {total} images",
			f"skipped existing {skipped_existing}",
			f"low quality {skipped_low_quality}",
		]
		if fusion_settings.enabled:
			summary_parts.append(f"fused {fused_created}")
		print(f"[info] Template cache updated: {', '.join(summary_parts)} into {templates_dir}")
		if skipped_low_quality:
			print(
				f"[info] {skipped_low_quality} images were rejected for quality below {args.quality_threshold:.2f}."
			)
		if args.probe is None:
			print("[info] Cache build finished. Provide --probe to run identification/verification.")
			return

	if args.probe is None:
		raise SystemExit("No probe fingerprint provided. Use --probe <image> to run matching.")

	try:
		gallery_templates = load_templates_from_directory(
			templates_dir,
			prefer_fused=not args.no_fusion,
			include_raw_when_fused=fusion_settings.keep_raw,
		)
	except FileNotFoundError as exc:
		raise SystemExit(str(exc)) from exc

	if not gallery_templates:
		raise SystemExit(
			f"No templates found in {templates_dir}. Build them first with --build-from <folder>."
		)
	low_quality_cached = [t for t in gallery_templates if t.quality < args.quality_threshold]
	if low_quality_cached:
		print(
			f"[warning] Ignoring {len(low_quality_cached)} cached templates below quality threshold "
			f"{args.quality_threshold:.2f}. Rebuild them if needed."
		)
	gallery_templates = [t for t in gallery_templates if t.quality >= args.quality_threshold]
	if not gallery_templates:
		raise SystemExit(
			f"No templates meet the quality threshold {args.quality_threshold:.2f}. Rebuild with "
			"--build-from using higher quality images."
		)
	if fusion_settings.enabled and not args.no_fusion and not any(t.fused for t in gallery_templates):
		fused_gallery = fuse_identity_templates(gallery_templates, hasher, fusion_settings)
		if fused_gallery:
			gallery_templates = fused_gallery

	probe_identifier = infer_identity_from_name(args.probe)
	probe_template = pipeline.process(args.probe, probe_identifier)
	if probe_template.quality < args.quality_threshold:
		raise SystemExit(
			f"Probe fingerprint quality {probe_template.quality:.3f} below threshold "
			f"{args.quality_threshold:.3f}. Capture another image."
		)
	print(f"[info] Probe quality: {probe_template.quality:.3f}")

	if args.mode == "verify":
		if not args.claim:
			raise SystemExit("--claim is required in verify mode.")
		claim_candidates = [t for t in gallery_templates if t.identifier == args.claim]
		if not claim_candidates:
			raise SystemExit(
				f"No templates for claimed identity '{args.claim}' meet the quality threshold "
				f"{args.quality_threshold:.2f}. Choose another image or rebuild the gallery."
			)

	matcher = FingerprintMatcher(gallery_templates, hasher=hasher)

	if args.mode == "identify":
		try:
			results = matcher.identify(probe_template, top_k=args.top, min_probe_quality=args.quality_threshold)
		except ValueError as e:
			print(f"[error] {e}")
			return
		
		if not results:
			print("No matches produced a valid score.")
			return
		print("Top matches:")
		for rank, (identity, score) in enumerate(results, start=1):
			print(f"{rank:2d}. {identity:15s} | score={score:.4f}")
	else:
		success, score = matcher.verify(probe_template, args.claim)
		label = "ACCEPT" if success else "REJECT"
		print(f"Verification result: {label} (score={score:.4f})")


if __name__ == "__main__":
	main()

