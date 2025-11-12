"""Data structures for Supermatcher v1.0 (Hybrid)

This module defines the core data classes used throughout the fingerprint matching system.
These classes are shared across all modules (preprocessing, extractor, template_creation, matching).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import numpy as np


@dataclass
class Minutia:
    """Fingerprint minutia (ridge ending or bifurcation).
    
    Attributes:
        x: X coordinate (pixels)
        y: Y coordinate (pixels)
        angle: Ridge direction in radians [0, 2π)
        kind: Minutia type ("ending" or "bifurcation")
        quality: Quality score [0.0, 1.0]
    """
    x: float
    y: float
    angle: float  # radians, ridge direction
    kind: str     # "ending" or "bifurcation"
    quality: float


@dataclass
class Pore:
    """Level-3 feature: sweat pore.
    
    Attributes:
        x: X coordinate (pixels)
        y: Y coordinate (pixels)
        radius: Pore radius (pixels)
        strength: Detection strength/confidence [0.0, 1.0]
    """
    x: float
    y: float
    radius: float
    strength: float


@dataclass
class FingerprintTemplate:
    """Complete fingerprint template with protected and raw features.
    
    This is the main data structure for storing processed fingerprint data.
    It contains both the cancelable (protected) template and optional raw features
    for geometric matching and fusion.
    
    Attributes:
        identifier: Unique identifier (e.g., user ID)
        image_path: Path to original fingerprint image
        protected: Packed cancelable template (numpy array of bits)
        bit_length: Length of protected template in bits
        quality: Overall quality score [0.0, 1.0]
        raw_features: Optional raw feature vector (for geometric matching)
        minutiae: Optional list of extracted minutiae
        fused: Whether this is a fused master template
        source_count: Number of samples used to create this template
        consensus_score: Consensus quality for fused templates [0.0, 1.0]
    """
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
        """Validate template after initialization."""
        if self.quality < 0.0 or self.quality > 1.0:
            raise ValueError(f"Quality must be in [0.0, 1.0], got {self.quality}")
        
        if self.consensus_score < 0.0 or self.consensus_score > 1.0:
            raise ValueError(f"Consensus score must be in [0.0, 1.0], got {self.consensus_score}")
        
        if self.source_count < 1:
            raise ValueError(f"Source count must be >= 1, got {self.source_count}")
        
        if self.bit_length <= 0:
            raise ValueError(f"Bit length must be positive, got {self.bit_length}")


@dataclass
class FusionSettings:
    """Configuration for template fusion.
    
    These settings control how multiple templates from the same user are fused
    into a single master template.
    
    Attributes:
        enabled: Whether fusion is enabled
        distance: Spatial threshold for minutiae consensus (pixels)
        angle_deg: Angular threshold for minutiae consensus (degrees)
        min_consensus: Minimum consensus ratio [0.0, 1.0]
        keep_raw: Whether to keep raw templates after fusion
        mode: Fusion mode ("optimal" or other)
    """
    enabled: bool = True
    distance: float = 12.0
    angle_deg: float = 15.0
    min_consensus: float = 0.5
    keep_raw: bool = False
    mode: str = "optimal"

    def __post_init__(self) -> None:
        """Validate fusion settings after initialization."""
        if self.distance <= 0:
            raise ValueError(f"Distance threshold must be positive, got {self.distance}")
        
        if self.angle_deg < 0 or self.angle_deg > 180:
            raise ValueError(f"Angle threshold must be in [0, 180] degrees, got {self.angle_deg}")
        
        if self.min_consensus < 0.0 or self.min_consensus > 1.0:
            raise ValueError(f"Min consensus must be in [0.0, 1.0], got {self.min_consensus}")
