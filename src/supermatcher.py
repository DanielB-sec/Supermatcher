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
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

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
	minutiae: List[Minutia]
	mask: np.ndarray
	orientation: np.ndarray  # radians, per block
	frequency: np.ndarray    # cycles / pixel, per block
	skeleton: np.ndarray     # thinned binary image
	level3: List[Pore]
	triangle_signatures: List[Tuple[float, float, float]]


# ---------------------------------------------------------------------------
# Constants and configuration defaults


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = SCRIPT_DIR / "fingerprints"


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
MATCH_MIN_SCORE = 0.12

# Level 3 (optional)
PORE_MIN_SIGMA = 0.6
PORE_MAX_SIGMA = 1.4
PORE_SIGMA_STEPS = 5
PORE_RESPONSE_THRESHOLD = 0.025


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
# Matching utilities


def match_minutiae(probe: FingerprintTemplate,
				   candidate: FingerprintTemplate,
				   distance_threshold: float = MATCH_DISTANCE_THRESHOLD,
				   angle_threshold: float = MATCH_ANGLE_THRESHOLD_RAD) -> float:
	"""Return similarity score between two templates based on minutiae alignment."""
	matches = 0
	used = set()

	for m in probe.minutiae:
		best_index = -1
		best_distance = float("inf")
		for idx, n in enumerate(candidate.minutiae):
			if idx in used:
				continue
			if m.kind != n.kind:
				continue
			dx = m.x - n.x
			dy = m.y - n.y
			distance = math.hypot(dx, dy)
			if distance > distance_threshold:
				continue
			angle_diff = abs(normalise_angle(m.angle - n.angle))
			if angle_diff > angle_threshold:
				continue
			if distance < best_distance:
				best_distance = distance
				best_index = idx
		if best_index >= 0:
			used.add(best_index)
			matches += 1

	denom = max(len(probe.minutiae), len(candidate.minutiae), 1)
	return matches / denom


def normalise_angle(angle: float) -> float:
	return (angle + math.pi) % (2.0 * math.pi) - math.pi


def match_level3_features(probe: FingerprintTemplate,
						  candidate: FingerprintTemplate) -> float:
	if not probe.level3 or not candidate.level3:
		return 0.0
	used = set()
	matches = 0
	for pore in probe.level3:
		best_idx = -1
		best_distance = float("inf")
		for idx, cand in enumerate(candidate.level3):
			if idx in used:
				continue
			dx = pore.x - cand.x
			dy = pore.y - cand.y
			distance = math.hypot(dx, dy)
			if distance > 12.0:
				continue
			radius_diff = abs(pore.radius - cand.radius)
			if radius_diff > 0.6:
				continue
			if distance < best_distance:
				best_distance = distance
				best_idx = idx
		if best_idx >= 0:
			used.add(best_idx)
			matches += 1
	denom = max(len(probe.level3), len(candidate.level3), 1)
	return matches / denom


def compare_triangle_signatures(probe: FingerprintTemplate,
								candidate: FingerprintTemplate) -> float:
	if not probe.triangle_signatures or not candidate.triangle_signatures:
		return 0.0
	set_probe = set(round_tuple(sig, 3) for sig in probe.triangle_signatures)
	set_candidate = set(round_tuple(sig, 3) for sig in candidate.triangle_signatures)
	intersection = set_probe & set_candidate
	denom = max(len(set_probe), len(set_candidate), 1)
	return len(intersection) / denom


def round_tuple(values: Tuple[float, ...], precision: int) -> Tuple[float, ...]:
	return tuple(round(v, precision) for v in values)


# ---------------------------------------------------------------------------
# Pipeline orchestrator


class FingerprintPipeline:
	def __init__(self, include_level3: bool = False) -> None:
		self.include_level3 = include_level3

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

		signatures = compute_delaunay_signatures(minutiae, skeleton.shape)

		return FingerprintTemplate(
			identifier=identifier,
			image_path=image_path,
			minutiae=minutiae,
			mask=mask.astype(np.uint8),
			orientation=orientation,
			frequency=frequency,
			skeleton=skeleton,
			level3=level3_features,
			triangle_signatures=signatures,
		)


# ---------------------------------------------------------------------------
# Matching engine


class FingerprintMatcher:
	def __init__(self, templates: Sequence[FingerprintTemplate], include_level3: bool) -> None:
		self.templates = list(templates)
		self.include_level3 = include_level3

	def identify(self, probe: FingerprintTemplate, top_k: int = 5) -> List[Tuple[str, float]]:
		scores: List[Tuple[str, float]] = []
		for candidate in self.templates:
			if candidate.image_path == probe.image_path:
				continue
			score_lvl2 = match_minutiae(probe, candidate)
			if self.include_level3:
				score_lvl3 = match_level3_features(probe, candidate)
				score_tri = compare_triangle_signatures(probe, candidate)
				score = 0.6 * score_lvl2 + 0.25 * score_lvl3 + 0.15 * score_tri
			else:
				score = score_lvl2
			scores.append((candidate.identifier, score))

		scores.sort(key=lambda item: item[1], reverse=True)
		return scores[:top_k]

	def verify(self, probe: FingerprintTemplate, claimed_id: str) -> Tuple[bool, float]:
		candidates = [t for t in self.templates if t.identifier == claimed_id]
		if not candidates:
			raise ValueError(f"No templates available for claimed identity '{claimed_id}'.")
		best_score = 0.0
		for candidate in candidates:
			score_lvl2 = match_minutiae(probe, candidate)
			if self.include_level3:
				score_lvl3 = match_level3_features(probe, candidate)
				score_tri = compare_triangle_signatures(probe, candidate)
				score = 0.6 * score_lvl2 + 0.25 * score_lvl3 + 0.15 * score_tri
			else:
				score = score_lvl2
			best_score = max(best_score, score)
		return best_score >= MATCH_MIN_SCORE, best_score


# ---------------------------------------------------------------------------
# Dataset helpers


def enumerate_database(db_path: Path) -> List[Path]:
	if not db_path.exists() or not db_path.is_dir():
		raise FileNotFoundError(f"Fingerprint database not found: {db_path}")
	files = [p for p in db_path.iterdir() if p.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}]
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
	parser.add_argument("--probe", type=Path, required=True, help="Fingerprint image to analyse.")
	parser.add_argument("--mode", choices=["identify", "verify"], default="identify", help="Operation mode.")
	parser.add_argument("--claim", type=str, default=None, help="Claimed identity for verification mode.")
	parser.add_argument("--top", type=int, default=5, help="Number of top matches to return in identification mode.")
	parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path to fingerprint gallery (default: ./fingerprints).")
	parser.add_argument("--use-level3", action="store_true", help="Enable Level-3 feature fusion (requires high-resolution images).")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.mode == "verify" and not args.claim:
		raise SystemExit("--claim is required in verify mode.")

	db_path = args.db
	gallery_paths = enumerate_database(db_path)

	pipeline = FingerprintPipeline(include_level3=args.use_level3)
	gallery_templates = build_gallery(pipeline, gallery_paths)

	probe_identifier = infer_identity_from_name(args.probe)
	probe_template = pipeline.process(args.probe, probe_identifier)

	matcher = FingerprintMatcher(gallery_templates, include_level3=args.use_level3)

	if args.mode == "identify":
		results = matcher.identify(probe_template, top_k=args.top)
		print("Top matches:")
		for rank, (identity, score) in enumerate(results, start=1):
			print(f"{rank:2d}. {identity:15s} | score={score:.4f}")
	else:
		success, score = matcher.verify(probe_template, args.claim)
		label = "ACCEPT" if success else "REJECT"
		print(f"Verification result: {label} (score={score:.4f})")


if __name__ == "__main__":
	main()

