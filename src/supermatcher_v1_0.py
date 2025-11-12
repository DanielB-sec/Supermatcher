"""Supermatcher v1.0 - Hybrid Fingerprint Identification/Authentication Pipeline

ARCHITECTURE:
- Modular design with 6 specialized modules:
  * config.py: Centralized configuration
  * dataclasses.py: Core data structures
  * preprocessing.py: Image processing (normalization, segmentation, diffusion)
  * extractor.py: Feature extraction (orientation, frequency, minutiae, pores)
  * template_creation.py: Hasher, fusion, quality assessment, I/O
  * matching.py: Two-stage identification + verification

PIPELINE:
1. Preprocessing: Normalization → Segmentation → Coherence Diffusion
2. Extraction: Orientation/Frequency → Log-Gabor Enhancement → Minutiae → Pores
3. Template Creation: Feature Vector → Quality Assessment → Protected Hash
4. Matching: Hash-based (Stage 1) → Geometric Reranking (Stage 2)

FEATURES:
- Preprocessing + extraction (robust enhancement)
- Identify/verify logic (geometric reranking)
- Cancelable templates (random projection + binarization)
- Template fusion (multi-sample enrollment)
- Adaptive thresholds (quality-based)

"""

from __future__ import annotations
import math
import os
import pickle
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Import all modules
from src.config import (
    HASH_FEATURE_DIM, HASH_PROJECTION_DIM, HASH_KEY, HASH_COUNT,
    DEFAULT_QUALITY_THRESHOLD, TEMPLATE_EXTENSION, DEFAULT_TEMPLATE_PATH,
    IMAGE_EXTENSIONS, PCORES_AFFINITY
)
from src.models import FingerprintTemplate, FusionSettings
from src.preprocessing import (
    load_grayscale_image, normalise_image, block_variance_segmentation, coherence_diffusion
)
from src.extractor import (
    estimate_orientation_and_frequency, apply_log_gabor_enhancement,
    binarise_and_thin, extract_minutiae, validate_minutiae, detect_pores
)
from src.template_creation import (
    CancelableHasher, evaluate_quality, build_feature_vector,
    save_template, load_templates_from_directory, fuse_identity_templates
)
from src.matching import FingerprintMatcher


# ---------------------------------------------------------------------------
# CPU Affinity Management


def set_cpu_affinity():
    """Set CPU affinity to P-cores only (if configured)."""
    if not PCORES_AFFINITY:
        return
    
    try:
        if hasattr(os, 'sched_setaffinity'):
            os.sched_setaffinity(0, PCORES_AFFINITY)
            print(f"[affinity] Set to P-cores: {PCORES_AFFINITY}")
    except (AttributeError, OSError) as e:
        print(f"[affinity] Could not set affinity: {e}")


# ---------------------------------------------------------------------------
# FingerprintPipeline: End-to-End Processing


class FingerprintPipeline:
    """End-to-end fingerprint processing pipeline.
    
    Processes raw fingerprint image through all stages:
    1. Preprocessing (load → normalize → segment → diffuse)
    2. Extraction (orientation/frequency → Log-Gabor → minutiae → pores)
    3. Template creation (feature vector → quality → protected hash)
    
    Attributes:
        hasher: CancelableHasher for protected template generation
        include_level3: Whether to extract Level-3 features (pores)
    """
    
    def __init__(
        self,
        hasher: Optional[CancelableHasher] = None,
        include_level3: bool = False
    ) -> None:
        """Initialize pipeline.
        
        Args:
            hasher: CancelableHasher instance (creates default if None)
            include_level3: Enable Level-3 feature extraction (default False)
        """
        if hasher is None:
            hasher = CancelableHasher(
                feature_dim=HASH_FEATURE_DIM,
                projection_dim=HASH_PROJECTION_DIM,
                key=HASH_KEY,
                hash_count=HASH_COUNT
            )
        
        self.hasher = hasher
        self.include_level3 = include_level3
    
    def process(
        self,
        image_path: Path,
        identifier: str,
        *,
        verbose: bool = False
    ) -> FingerprintTemplate:
        """Process fingerprint image into template.
        
        Args:
            image_path: Path to fingerprint image
            identifier: User identifier
            verbose: Print processing steps (default False)
            
        Returns:
            FingerprintTemplate with protected hash, minutiae, quality score
            
        Raises:
            FileNotFoundError: If image_path does not exist
            ValueError: If image processing fails
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if verbose:
            print(f"[process] Processing {image_path.name}...")
        
        # Phase 1: Preprocessing
        image = load_grayscale_image(image_path)
        normalised = normalise_image(image)
        mask = block_variance_segmentation(normalised)
        diffused = coherence_diffusion(normalised, mask)
        
        if verbose:
            print(f"[process]   Preprocessing complete (mask coverage: {mask.mean():.2%})")
        
        # Phase 2: Feature extraction
        orientation, frequency = estimate_orientation_and_frequency(diffused, mask)
        enhanced = apply_log_gabor_enhancement(diffused, mask, orientation, frequency)
        binary, skeleton = binarise_and_thin(enhanced, mask)
        
        minutiae = extract_minutiae(skeleton, mask, orientation)
        minutiae = validate_minutiae(minutiae, skeleton, mask)
        
        level3_features = []
        if self.include_level3:
            level3_features = detect_pores(enhanced, mask)
        
        if verbose:
            print(f"[process]   Extracted {len(minutiae)} minutiae" +
                  (f", {len(level3_features)} pores" if level3_features else ""))
        
        # Phase 3: Template creation
        quality_score = evaluate_quality(
            minutiae, mask, orientation, frequency, skeleton, enhanced, diffused,
            level3=level3_features if self.include_level3 else None
        )
        
        feature_vector = build_feature_vector(
            minutiae, mask, orientation, frequency, skeleton, enhanced, diffused,
            level3=level3_features if self.include_level3 else None,
            use_consensus_weighting=True
        )
        
        protected = self.hasher.encode(feature_vector)
        
        if verbose:
            print(f"[process]   Quality: {quality_score:.3f}")
        
        return FingerprintTemplate(
            identifier=identifier,
            image_path=image_path,
            protected=protected,
            bit_length=self.hasher.bit_length,
            quality=quality_score,
            raw_features=feature_vector.astype(np.float32, copy=True),
            minutiae=list(minutiae),
            consensus_score=1.0
        )


# ---------------------------------------------------------------------------
# High-Level API: Enroll, Identify, Verify


def enroll(
    image_paths: Sequence[Path],
    identifier: str,
    output_dir: Path = None,
    *,
    fusion_settings: Optional[FusionSettings] = None,
    quality_threshold: float = None,
    include_level3: bool = False,
    verbose: bool = True
) -> List[FingerprintTemplate]:
    """Enroll user by processing multiple fingerprint samples.
    
    Workflow:
    1. Process all images for given identifier
    2. Filter by quality threshold
    3. Optionally fuse multiple samples into master template
    4. Save templates to output_dir
    
    Args:
        image_paths: List of fingerprint image paths for same user
        identifier: User identifier (e.g., "101", "john_doe")
        output_dir: Directory to save templates (uses default if None)
        fusion_settings: Fusion configuration (None = no fusion)
        quality_threshold: Minimum quality to accept (uses default if None)
        include_level3: Extract Level-3 features (default False)
        verbose: Print progress messages (default True)
        
    Returns:
        List of FingerprintTemplate objects (fused + raw if keep_raw=True)
    """
    if output_dir is None:
        output_dir = DEFAULT_TEMPLATE_PATH
    
    if quality_threshold is None:
        quality_threshold = DEFAULT_QUALITY_THRESHOLD
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n[enroll] Enrolling user '{identifier}' with {len(image_paths)} samples")
    
    # Create pipeline
    pipeline = FingerprintPipeline(include_level3=include_level3)
    
    # Process all images
    templates: List[FingerprintTemplate] = []
    for image_path in image_paths:
        try:
            template = pipeline.process(image_path, identifier, verbose=verbose)
            
            if template.quality >= quality_threshold:
                templates.append(template)
                if verbose:
                    print(f"[enroll]   ✓ {image_path.name}: quality={template.quality:.3f}")
            else:
                if verbose:
                    print(f"[enroll]   ✗ {image_path.name}: quality={template.quality:.3f} "
                          f"(below threshold {quality_threshold})")
        except Exception as e:
            if verbose:
                print(f"[enroll]   ERROR processing {image_path.name}: {e}")
    
    if not templates:
        if verbose:
            print(f"[enroll] No valid templates for '{identifier}'")
        return []
    
    # Apply fusion if enabled
    if fusion_settings is not None and fusion_settings.enabled and len(templates) > 1:
        if verbose:
            print(f"[enroll] Fusing {len(templates)} templates...")
        
        fused_templates = fuse_identity_templates(templates, pipeline.hasher, fusion_settings)
        templates = fused_templates
        
        if verbose:
            fused_count = sum(1 for t in templates if t.fused)
            print(f"[enroll] Created {fused_count} fused template(s)")
    
    # Save templates
    saved_count = 0
    for template in templates:
        try:
            save_template(template, output_dir, overwrite=True)
            saved_count += 1
        except Exception as e:
            if verbose:
                print(f"[enroll] ERROR saving template: {e}")
    
    if verbose:
        print(f"[enroll] Saved {saved_count} template(s) to {output_dir}")
    
    return templates


def identify(
    probe_path: Path,
    gallery_dir: Path = None,
    *,
    top_k: int = 5,
    use_geometric_reranking: bool = True,
    quality_threshold: float = None,
    prefer_fused: bool = True,
    include_level3: bool = False,
    verbose: bool = True
) -> List[Tuple[str, float]]:
    """Identify probe fingerprint against gallery (1:N matching).
    
    Workflow:
    1. Process probe image
    2. Load gallery templates
    3. Use FingerprintMatcher.identify() with geometric reranking
    4. Return top-K matches
    
    Args:
        probe_path: Path to probe fingerprint image
        gallery_dir: Directory containing gallery templates (uses default if None)
        top_k: Number of top matches to return (default 5)
        use_geometric_reranking: Enable Stage 2 geometric matching (default True)
        quality_threshold: Minimum probe quality (uses default if None)
        prefer_fused: Prefer fused templates over raw (default True)
        include_level3: Extract Level-3 features (default False)
        verbose: Print matching details (default True)
        
    Returns:
        List of (identifier, score) tuples sorted by score (descending)
        
    Raises:
        ValueError: If probe quality is below threshold
        FileNotFoundError: If gallery_dir contains no templates
    """
    if gallery_dir is None:
        gallery_dir = DEFAULT_TEMPLATE_PATH
    
    if quality_threshold is None:
        quality_threshold = DEFAULT_QUALITY_THRESHOLD
    
    gallery_dir = Path(gallery_dir)
    
    if verbose:
        print(f"\n[identify] Processing probe: {probe_path.name}")
    
    # Process probe
    pipeline = FingerprintPipeline(include_level3=include_level3)
    probe = pipeline.process(probe_path, identifier=probe_path.stem, verbose=verbose)
    
    if probe.quality < quality_threshold:
        raise ValueError(
            f"Probe quality {probe.quality:.3f} is below threshold {quality_threshold}. "
            f"Please recapture the fingerprint."
        )
    
    # Load gallery
    if verbose:
        print(f"[identify] Loading gallery from {gallery_dir}")
    
    gallery_templates = load_templates_from_directory(
        gallery_dir,
        prefer_fused=prefer_fused,
        include_raw_when_fused=False
    )
    
    if not gallery_templates:
        raise FileNotFoundError(f"No templates found in {gallery_dir}")
    
    if verbose:
        print(f"[identify] Gallery: {len(gallery_templates)} templates")
    
    # Create matcher and identify
    matcher = FingerprintMatcher(gallery_templates, pipeline.hasher)
    results = matcher.identify(
        probe,
        top_k=top_k,
        use_geometric_reranking=use_geometric_reranking,
        min_probe_quality=quality_threshold
    )
    
    if verbose:
        print(f"\n[identify] Top-{min(len(results), top_k)} matches:")
        for i, (identity, score) in enumerate(results[:top_k], 1):
            print(f"  {i}. {identity}: {score:.4f}")
    
    return results


def verify(
    probe_path: Path,
    claimed_id: str,
    gallery_dir: Path = None,
    *,
    quality_threshold: float = None,
    use_adaptive_threshold: bool = True,
    include_level3: bool = False,
    verbose: bool = True
) -> Tuple[bool, float]:
    """Verify probe fingerprint against claimed identity (1:1 matching).
    
    Workflow:
    1. Process probe image
    2. Load gallery templates for claimed_id
    3. Use FingerprintMatcher.verify() with adaptive threshold
    4. Return (is_match, score)
    
    Args:
        probe_path: Path to probe fingerprint image
        claimed_id: Claimed user identifier
        gallery_dir: Directory containing gallery templates (uses default if None)
        quality_threshold: Minimum probe quality (uses default if None)
        use_adaptive_threshold: Use quality-based threshold (default True)
        include_level3: Extract Level-3 features (default False)
        verbose: Print verification details (default True)
        
    Returns:
        Tuple of (is_match, score):
        - is_match: True if score ≥ threshold
        - score: Best similarity score
        
    Raises:
        ValueError: If probe quality is below threshold or no templates for claimed_id
    """
    if gallery_dir is None:
        gallery_dir = DEFAULT_TEMPLATE_PATH
    
    if quality_threshold is None:
        quality_threshold = DEFAULT_QUALITY_THRESHOLD
    
    gallery_dir = Path(gallery_dir)
    
    if verbose:
        print(f"\n[verify] Processing probe: {probe_path.name}")
        print(f"[verify] Claimed identity: {claimed_id}")
    
    # Process probe
    pipeline = FingerprintPipeline(include_level3=include_level3)
    probe = pipeline.process(probe_path, identifier=probe_path.stem, verbose=verbose)
    
    if probe.quality < quality_threshold:
        raise ValueError(
            f"Probe quality {probe.quality:.3f} is below threshold {quality_threshold}. "
            f"Please recapture the fingerprint."
        )
    
    # Load gallery
    gallery_templates = load_templates_from_directory(gallery_dir)
    
    if not gallery_templates:
        raise FileNotFoundError(f"No templates found in {gallery_dir}")
    
    # Create matcher and verify
    matcher = FingerprintMatcher(gallery_templates, pipeline.hasher)
    is_match, score = matcher.verify(
        probe,
        claimed_id,
        use_adaptive_threshold=use_adaptive_threshold
    )
    
    if verbose:
        print(f"\n[verify] Result: {'✓ MATCH' if is_match else '✗ NO MATCH'}")
        print(f"[verify] Score: {score:.4f}")
    
    return is_match, score


# ---------------------------------------------------------------------------
# Utility Functions


def enumerate_database(db_path: Path) -> List[Path]:
    """List all fingerprint images in database directory.
    
    Args:
        db_path: Path to fingerprint database directory
        
    Returns:
        List of image file paths (sorted)
        
    Raises:
        FileNotFoundError: If db_path doesn't exist or contains no images
    """
    if not db_path.exists() or not db_path.is_dir():
        raise FileNotFoundError(f"Database directory not found: {db_path}")
    
    files = [p for p in db_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    
    if not files:
        raise FileNotFoundError(f"No fingerprint images found in {db_path}")
    
    files.sort()
    return files


def infer_identity_from_filename(path: Path) -> str:
    """Infer user identifier from filename (e.g., "101_1.tif" → "101").
    
    Args:
        path: Image file path
        
    Returns:
        User identifier (string before first underscore, or full stem if no underscore)
    """
    stem = path.stem
    if "_" in stem:
        return stem.split("_")[0]
    return stem


# ---------------------------------------------------------------------------
# Main Entry Point


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Supermatcher v1.0 - Hybrid Fingerprint Identification/Authentication"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Enroll command
    enroll_parser = subparsers.add_parser("enroll", help="Enroll user fingerprints")
    enroll_parser.add_argument("identifier", type=str, help="User identifier")
    enroll_parser.add_argument("images", type=Path, nargs="+", help="Fingerprint images")
    enroll_parser.add_argument("--output", "-o", type=Path, default=None, help="Output directory")
    enroll_parser.add_argument("--fusion", action="store_true", help="Enable template fusion")
    enroll_parser.add_argument("--level3", action="store_true", help="Extract Level-3 features")
    
    # Identify command
    identify_parser = subparsers.add_parser("identify", help="Identify probe (1:N)")
    identify_parser.add_argument("probe", type=Path, help="Probe fingerprint image")
    identify_parser.add_argument("--gallery", "-g", type=Path, default=None, help="Gallery directory")
    identify_parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of top matches")
    identify_parser.add_argument("--no-geometric", action="store_true", help="Disable geometric reranking")
    identify_parser.add_argument("--level3", action="store_true", help="Extract Level-3 features")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify probe (1:1)")
    verify_parser.add_argument("probe", type=Path, help="Probe fingerprint image")
    verify_parser.add_argument("claimed_id", type=str, help="Claimed identity")
    verify_parser.add_argument("--gallery", "-g", type=Path, default=None, help="Gallery directory")
    verify_parser.add_argument("--level3", action="store_true", help="Extract Level-3 features")
    
    args = parser.parse_args()
    
    # Set CPU affinity
    set_cpu_affinity()
    
    if args.command == "enroll":
        fusion_settings = None
        if args.fusion:
            fusion_settings = FusionSettings(
                enabled=True,
                distance=12.0,
                angle_deg=20.0,
                min_consensus=0.4,
                keep_raw=False,
                mode="optimal"
            )
        
        enroll(
            image_paths=args.images,
            identifier=args.identifier,
            output_dir=args.output,
            fusion_settings=fusion_settings,
            include_level3=args.level3,
            verbose=True
        )
    
    elif args.command == "identify":
        results = identify(
            probe_path=args.probe,
            gallery_dir=args.gallery,
            top_k=args.top_k,
            use_geometric_reranking=not args.no_geometric,
            include_level3=args.level3,
            verbose=True
        )
    
    elif args.command == "verify":
        is_match, score = verify(
            probe_path=args.probe,
            claimed_id=args.claimed_id,
            gallery_dir=args.gallery,
            include_level3=args.level3,
            verbose=True
        )
    
    else:
        parser.print_help()
