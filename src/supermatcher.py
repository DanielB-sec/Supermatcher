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

	def __post_init__(self) -> None:
		self.protected = np.asarray(self.protected, dtype=np.uint8)
		if self.protected.ndim != 1:
			raise ValueError("Protected template must be a 1-D array of packed bits.")
		expected_len = (self.bit_length + 7) // 8
		if self.protected.size != expected_len:
			raise ValueError(
				f"Protected template has {self.protected.size} bytes but expected {expected_len}."
			)


# ---------------------------------------------------------------------------
# Constants and configuration defaults


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = SCRIPT_DIR / "fingerprints"
DEFAULT_TEMPLATE_PATH = SCRIPT_DIR / "templates"

IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
TEMPLATE_EXTENSION = ".fpt"


# Phase 1 parameters
NORMALISATION_BLOCK = 16
NORMALISATION_MEAN = 100.0
NORMALISATION_VAR = 100.0

SEGMENT_BLOCK = 16
SEGMENT_THRESHOLD = 70.0


# Phase 2 parameters
CED_ITERATIONS = 12
CED_DELTA_T = 0.15
CED_TENSOR_SIGMA = 2.0
CED_GRAD_SIGMA = 1.0
CED_ALPHA = 0.01
CED_BETA = 1.25

ORIENTATION_BLOCK = 16
LOG_GABOR_SCALES = (0.85, 1.0, 1.2)
LOG_GABOR_SIGMA_R = 1.5
LOG_GABOR_SIGMA_THETA_DEG = 12.0


# Phase 3 parameters
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
	image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
	if image is None:
		raise FileNotFoundError(f"Unable to read fingerprint image: {path}")
	return image.astype(np.float32)


def normalise_image(image: np.ndarray,
					block_size: int = NORMALISATION_BLOCK,
					mean0: float = NORMALISATION_MEAN,
					var0: float = NORMALISATION_VAR) -> np.ndarray:
	"""Normalise intensity using local statistics to enforce desired mean/variance."""
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
						beta: float = CED_BETA) -> np.ndarray:
	"""Apply coherence-enhancing diffusion following Weickert's formulation."""

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
					  mask: np.ndarray) -> List[Minutia]:
	"""Remove spurious minutiae based on spatial heuristics."""
	if not minutiae:
		return []

	filtered: List[Minutia] = []
	for m in minutiae:
		if not mask[int(m.y), int(m.x)]:
			continue
		filtered.append(m)

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


def load_templates_from_directory(directory: Path) -> List[FingerprintTemplate]:
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
	return templates


def build_template_cache(pipeline: FingerprintPipeline,
					 raw_dir: Path,
					 output_dir: Path,
					 *,
					 overwrite: bool = False) -> Tuple[int, int, int]:
	image_paths = enumerate_database(raw_dir)
	total = len(image_paths)
	processed = 0
	skipped = 0
	for path in image_paths:
		target = template_output_path(output_dir, path)
		if target.exists() and not overwrite:
			skipped += 1
			continue
		identifier = infer_identity_from_name(path)
		template = pipeline.process(path, identifier)
		save_template(template, output_dir, overwrite=True)
		processed += 1
	return processed, skipped, total


# ---------------------------------------------------------------------------
# Pipeline orchestrator


class FingerprintPipeline:
	def __init__(self, include_level3: bool = False, hasher: Optional[CancelableHasher] = None) -> None:
		self.include_level3 = include_level3
		if hasher is None:
			raise ValueError("CancelableHasher instance must be provided for protected templates.")
		self.hasher = hasher

	def process(self, image_path: Path, identifier: str) -> FingerprintTemplate:
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
		)


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

	def identify(self, probe: FingerprintTemplate, top_k: int = 5) -> List[Tuple[str, float]]:
		scores: List[Tuple[str, float]] = []
		for candidate in self.templates:
			if candidate.image_path == probe.image_path:
				continue
			similarity = self.hasher.similarity(probe.protected, candidate.protected)
			scores.append((candidate.image_path.name, similarity))

		scores.sort(key=lambda item: item[1], reverse=True)
		return scores[:top_k]

	def verify(self, probe: FingerprintTemplate, claimed_id: str) -> Tuple[bool, float]:
		candidates = [t for t in self.templates if t.identifier == claimed_id]
		if not candidates:
			raise ValueError(f"No templates available for claimed identity '{claimed_id}'.")
		best_score = 0.0
		for candidate in candidates:
			similarity = self.hasher.similarity(probe.protected, candidate.protected)
			best_score = max(best_score, similarity)
		return best_score >= MATCH_MIN_SCORE, best_score


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


def build_gallery(pipeline: FingerprintPipeline, image_paths: Sequence[Path]) -> List[FingerprintTemplate]:
	templates: List[FingerprintTemplate] = []
	for path in image_paths:
		identifier = infer_identity_from_name(path)
		template = pipeline.process(path, identifier)
		templates.append(template)
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
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.projection_dim <= 0:
		raise SystemExit("--projection-dim must be positive.")
	if args.hash_count <= 0:
		raise SystemExit("--hash-count must be positive.")
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
			processed, skipped, total = build_template_cache(
				pipeline,
				raw_dir=args.build_from,
				output_dir=templates_dir,
				overwrite=args.overwrite_templates,
			)
		except FileNotFoundError as exc:
			raise SystemExit(str(exc)) from exc
		print(
			f"[info] Template cache updated: processed {processed} of {total} images "
			f"(skipped {skipped}) into {templates_dir}"
		)
		if args.probe is None:
			print("[info] Cache build finished. Provide --probe to run identification/verification.")
			return

	if args.probe is None:
		raise SystemExit("No probe fingerprint provided. Use --probe <image> to run matching.")

	if args.mode == "verify" and not args.claim:
		raise SystemExit("--claim is required in verify mode.")

	try:
		gallery_templates = load_templates_from_directory(templates_dir)
	except FileNotFoundError as exc:
		raise SystemExit(str(exc)) from exc

	if not gallery_templates:
		raise SystemExit(
			f"No templates found in {templates_dir}. Build them first with --build-from <folder>."
		)

	probe_identifier = infer_identity_from_name(args.probe)
	probe_template = pipeline.process(args.probe, probe_identifier)

	matcher = FingerprintMatcher(gallery_templates, hasher=hasher)

	if args.mode == "identify":
		results = matcher.identify(probe_template, top_k=args.top)
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

