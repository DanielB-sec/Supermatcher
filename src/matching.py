"""Matching module for Supermatcher v1.0 (Hybrid)

This module contains:
- compute_geometric_minutiae_score: RANSAC-based minutiae matching
- FingerprintMatcher: Two-stage identification (hash + geometric reranking) and verification
- identify: 1:N identification with adaptive thresholds
- verify: 1:1 verification

"""

from __future__ import annotations
import math
from typing import List, Sequence, Tuple
import numpy as np
from scipy.spatial import KDTree

from src.models import Minutia, FingerprintTemplate
from src.template_creation import CancelableHasher
from src.config import (
    MATCH_DISTANCE_THRESHOLD, MATCH_ANGLE_THRESHOLD_RAD,
    MATCH_MIN_SCORE, GEOMETRIC_RERANKING_TIE_THRESHOLD,
    GEOMETRIC_RERANKING_SCORE_DIFF_THRESHOLD
)


# ---------------------------------------------------------------------------
# Geometric Minutiae Matching


def compute_geometric_minutiae_score(
    probe_minutiae: Sequence[Minutia],
    candidate_minutiae: Sequence[Minutia],
    distance_threshold: float = None,
    angle_threshold: float = None,
) -> float:
    """Compute geometric similarity score between two minutiae sets using RANSAC alignment.
    
    This function performs direct minutiae-to-minutiae matching using spatial and angular
    correspondence after rigid transformation alignment (rotation + translation).
    
    Algorithm:
    1. RANSAC to find best rigid transformation (rotation + translation)
    2. Transform probe minutiae to candidate coordinate system
    3. Count inliers (minutiae pairs within distance and angle thresholds)
    4. Normalize score by larger set size (penalizes size mismatch)
    
    Args:
        probe_minutiae: Probe minutiae list
        candidate_minutiae: Candidate minutiae list
        distance_threshold: Max distance for correspondence (uses config default if None)
        angle_threshold: Max angle difference for correspondence radians (uses config default if None)
        
    Returns:
        Geometric similarity score in [0, 1] range
    """
    if distance_threshold is None:
        distance_threshold = MATCH_DISTANCE_THRESHOLD
    if angle_threshold is None:
        angle_threshold = MATCH_ANGLE_THRESHOLD_RAD
    
    if not probe_minutiae or not candidate_minutiae:
        return 0.0
    
    n_probe = len(probe_minutiae)
    n_candidate = len(candidate_minutiae)
    
    probe_points = np.array([[m.x, m.y] for m in probe_minutiae], dtype=np.float32)
    cand_points = np.array([[m.x, m.y] for m in candidate_minutiae], dtype=np.float32)
    
    # Fallback for insufficient points
    if n_probe < 3 or n_candidate < 3:
        tree = KDTree(cand_points)
        distances, _ = tree.query(probe_points, k=1)
        matches = np.sum(distances < distance_threshold)
        return float(matches) / float(max(n_probe, n_candidate))
    
    # RANSAC to find best transformation
    best_inliers = 0
    best_transform = None
    max_iterations = min(100, n_probe * 2)
    
    for _ in range(max_iterations):
        if n_probe < 3:
            break
        
        sample_indices = np.random.choice(n_probe, size=3, replace=False)
        probe_sample = probe_points[sample_indices]
        
        tree = KDTree(cand_points)
        distances, cand_indices = tree.query(probe_sample, k=1)
        
        if np.any(distances > distance_threshold * 2):
            continue
        
        cand_sample = cand_points[cand_indices]
        
        try:
            # Compute rigid transformation using SVD
            probe_centroid = probe_sample.mean(axis=0)
            cand_centroid = cand_sample.mean(axis=0)
            
            probe_centered = probe_sample - probe_centroid
            cand_centered = cand_sample - cand_centroid
            
            H = probe_centered.T @ cand_centered
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            t = cand_centroid - R @ probe_centroid
            transformed = (R @ probe_points.T).T + t
            
            # Count inliers (spatial + angular)
            tree_full = KDTree(cand_points)
            distances_full, indices_full = tree_full.query(transformed, k=1)
            
            spatial_inliers = distances_full < distance_threshold
            rotation_angle = math.atan2(R[1, 0], R[0, 0])
            angular_inliers = np.zeros(n_probe, dtype=bool)
            
            for i, (is_spatial, cand_idx) in enumerate(zip(spatial_inliers, indices_full)):
                if not is_spatial:
                    continue
                
                probe_angle = probe_minutiae[i].angle
                cand_angle = candidate_minutiae[cand_idx].angle
                rotated_angle = (probe_angle + rotation_angle) % (2.0 * math.pi)
                
                angle_diff = abs(rotated_angle - cand_angle)
                angle_diff = min(angle_diff, 2.0 * math.pi - angle_diff)
                angular_inliers[i] = angle_diff < angle_threshold
            
            combined_inliers = spatial_inliers & angular_inliers
            inlier_count = np.sum(combined_inliers)
            
            if inlier_count > best_inliers:
                best_inliers = inlier_count
                best_transform = (R, t, rotation_angle)
        
        except (np.linalg.LinAlgError, ValueError):
            continue
    
    if best_transform is None or best_inliers < 3:
        # Fallback: simple nearest neighbor
        tree = KDTree(cand_points)
        distances, _ = tree.query(probe_points, k=1)
        matches = np.sum(distances < distance_threshold)
        score = float(matches) / float(max(n_probe, n_candidate))
        return score * 0.5  # Penalize (no good alignment)
    
    # Compute final score
    max_minutiae = max(n_probe, n_candidate)
    geometric_score = float(best_inliers) / float(max_minutiae)
    
    probe_ratio = float(best_inliers) / float(n_probe)
    candidate_ratio = float(best_inliers) / float(n_candidate)
    
    final_score = 0.6 * geometric_score + 0.2 * probe_ratio + 0.2 * candidate_ratio
    
    return float(np.clip(final_score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# FingerprintMatcher Class


class FingerprintMatcher:
    """Two-stage fingerprint matcher: hash-based + geometric reranking.
    
    Implements production-grade identification and verification:
    - Stage 1 (fast): Hash-based similarity (all candidates)
    - Stage 2 (accurate): Geometric minutiae matching (top candidates with close scores)
    
    Attributes:
        templates: List of gallery templates
        hasher: CancelableHasher instance for protected template comparison
    """
    
    def __init__(self, templates: Sequence[FingerprintTemplate], hasher: CancelableHasher) -> None:
        """Initialize matcher with gallery templates.
        
        Args:
            templates: Sequence of gallery templates
            hasher: CancelableHasher instance (must match templates' bit_length)
            
        Raises:
            ValueError: If template bit_length doesn't match hasher
        """
        self.templates = list(templates)
        self.hasher = hasher
        expected_bits = self.hasher.bit_length
        
        for template in self.templates:
            if template.bit_length != expected_bits:
                raise ValueError(
                    f"Template bit-length mismatch: expected {expected_bits}, got {template.bit_length}. "
                    f"Rebuild templates with current projection key."
                )
    
    def adaptive_threshold(self, probe_quality: float, candidate_quality: float) -> float:
        """Compute adaptive matching threshold based on template qualities.
        
        Strategy (tuned for production):
        - Both high quality (>0.7): Use standard threshold (0.78)
        - Both medium quality (≥0.5): Relax slightly (-0.03 → 0.75)
        - At least one low quality: Relax more (-0.05 → 0.73)
        
        Args:
            probe_quality: Probe template quality [0, 1]
            candidate_quality: Candidate template quality [0, 1]
            
        Returns:
            Adjusted matching threshold
        """
        if probe_quality > 0.7 and candidate_quality > 0.7:
            return MATCH_MIN_SCORE
        
        if probe_quality >= 0.5 and candidate_quality >= 0.5:
            return MATCH_MIN_SCORE - 0.03  # 0.75
        
        return MATCH_MIN_SCORE - 0.05  # 0.73
    
    def identify(
        self,
        probe: FingerprintTemplate,
        top_k: int = 5,
        use_geometric_reranking: bool = True,
        min_probe_quality: float = 0.35
    ) -> List[Tuple[str, float]]:
        """Two-stage identification: hash-based matching + geometric verification.
        
        Algorithm:
        1. Compute hash similarity for all gallery templates (Stage 1)
        2. If top scores are ambiguous (diff < 0.02), apply geometric reranking (Stage 2)
        3. Include tied candidates (score diff < 0.005) in reranking pool
        4. Rerank top-N using combined score: 60% hash + 40% geometric
        5. Return top-K matches with combined scores (reranked) or normalized scores (others)
        
        Args:
            probe: Probe fingerprint template
            top_k: Number of top matches to return (default 5)
            use_geometric_reranking: Enable geometric reranking (default True)
            min_probe_quality: Minimum acceptable probe quality (default 0.35)
            
        Returns:
            List of (identifier, score) tuples sorted by score (descending)
            - Reranked candidates: combined score (60% hash + 40% geometric)
            - Other candidates: normalized score (60% hash)
            
        Raises:
            ValueError: If probe quality is below min_probe_quality threshold
        """
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
            hash_scores.append((candidate.identifier, hash_similarity, candidate))
        
        hash_scores.sort(key=lambda item: item[1], reverse=True)
        
        # Early exit if reranking disabled or too few candidates
        if not use_geometric_reranking or len(hash_scores) < 2:
            return [(name, score) for name, score, _ in hash_scores[:top_k]]
        
        # Stage 2: Geometric verification for top candidates
        # EXPANDED: Start with TOP-5 instead of TOP-3 for better coverage
        top_candidates = hash_scores[:min(5, len(hash_scores))]
        
        # Detect ties: include candidates beyond position 5 with similar scores
        if len(hash_scores) > 5:
            fifth_score = hash_scores[4][1]
            tie_threshold = GEOMETRIC_RERANKING_TIE_THRESHOLD
            
            for i in range(5, min(len(hash_scores), 8)):
                score_diff = abs(hash_scores[i][1] - fifth_score)
                if score_diff < tie_threshold:
                    top_candidates.append(hash_scores[i])
                    print(f"[geometric] Including tied candidate at position {i+1}: {hash_scores[i][0]} "
                          f"(diff={score_diff:.6f})")
                else:
                    break
        
        # Check if reranking is needed (top scores are ambiguous)
        if len(top_candidates) >= 2:
            score_diff = top_candidates[0][1] - top_candidates[1][1]
            
            # AGGRESSIVE RERANKING: Increased threshold from 0.02 to 0.03
            if score_diff < GEOMETRIC_RERANKING_SCORE_DIFF_THRESHOLD:
                print(f"[geometric] Hash scores close (diff={score_diff:.4f}), "
                      f"applying geometric verification to top-{len(top_candidates)}...")
                
                # Compute geometric scores
                reranking_data: List[Tuple[str, float, float, float]] = []
                for name, hash_score, candidate in top_candidates:
                    geometric_score = compute_geometric_minutiae_score(
                        probe.minutiae,
                        candidate.minutiae,
                        distance_threshold=MATCH_DISTANCE_THRESHOLD,
                        angle_threshold=MATCH_ANGLE_THRESHOLD_RAD,
                    )

                    # Combined score: ADJUSTED 93% hash + 7% geometric (BOOST GEOMETRIC WEIGHT!)
                    combined_score = 0.9269 * hash_score + 0.0731 * geometric_score
                    
                    reranking_data.append((name, hash_score, geometric_score, combined_score))
                    print(f"[geometric]   {name}: hash={hash_score:.4f}, geom={geometric_score:.4f}, "
                          f"combined={combined_score:.4f}")
                
                # Re-rank by combined score
                reranking_data.sort(key=lambda item: item[3], reverse=True)
                
                # Build final result
                final_scores: List[Tuple[str, float]] = []
                reranked_names = set()
                
                # Add reranked candidates with combined scores
                for name, hash_score, geom_score, combined in reranking_data:
                    final_scores.append((name, combined))
                    reranked_names.add(name)
                    print(f"[geometric] Reranked: {name} (combined={combined:.4f})")
                
                # Add remaining candidates with normalized scores (0.5 * hash, matching weight)
                for name, hash_score, _ in hash_scores:
                    if name not in reranked_names:
                        normalized_score = 0.5 * hash_score  # Match 50% hash weight
                        final_scores.append((name, normalized_score))
                
                return final_scores[:top_k]
        
        # No reranking needed (scores not close)
        return [(name, score) for name, score, _ in hash_scores[:top_k]]
    
    def verify(
        self,
        probe: FingerprintTemplate,
        claimed_id: str,
        use_adaptive_threshold: bool = True
    ) -> Tuple[bool, float]:
        """1:1 verification: Check if probe matches claimed identity.
        
        Compares probe against all templates with claimed_id, returns best match score.
        
        Args:
            probe: Probe fingerprint template
            claimed_id: Claimed user identifier
            use_adaptive_threshold: Use quality-adaptive threshold (default True)
            
        Returns:
            Tuple of (is_match, best_score):
            - is_match: True if best_score ≥ threshold
            - best_score: Highest similarity score among claimed_id templates
            
        Raises:
            ValueError: If no templates exist for claimed_id
        """
        candidates = [t for t in self.templates if t.identifier == claimed_id]
        
        if not candidates:
            raise ValueError(f"No templates available for claimed identity '{claimed_id}'.")
        
        best_score = 0.0
        for candidate in candidates:
            similarity = self.hasher.similarity(probe.protected, candidate.protected)
            best_score = max(best_score, similarity)
        
        if use_adaptive_threshold:
            # Use average candidate quality for threshold adaptation
            avg_candidate_quality = sum(c.quality for c in candidates) / len(candidates)
            threshold = self.adaptive_threshold(probe.quality, avg_candidate_quality)
        else:
            threshold = MATCH_MIN_SCORE
        
        return best_score >= threshold, best_score
