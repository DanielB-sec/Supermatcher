"""Secure serialization for FingerprintTemplate (Protected + Minutiae only)

This module provides functions to serialize/deserialize fingerprint templates
in a SECURE way: only the cancelable hash (protected) and minutiae are persisted.

RAW FEATURES ARE NEVER STORED to prevent template inversion attacks.

Migration from pickle (insecure) to dict-based serialization.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import json

from src.models import Minutia, FingerprintTemplate


def template_to_secure_dict(template: FingerprintTemplate) -> Dict[str, Any]:
    """Serialize template to secure dictionary (Protected + Minutiae only).
    
    This function extracts ONLY the secure components of a template:
    - Protected hash (cancelable biometric)
    - Minutiae coordinates (reversible but limited information)
    - Metadata (quality, fusion info, etc.)
    
    RAW_FEATURES are NEVER included to prevent reconstruction attacks.
    
    Args:
        template: FingerprintTemplate object
        
    Returns:
        Dictionary with secure data only
    """
    return {
        'identifier': template.identifier,
        'image_path': str(template.image_path),
        'protected': template.protected.tobytes(),  # Cancelable hash as bytes
        'bit_length': template.bit_length,
        'quality': float(template.quality),
        'minutiae': [
            {
                'x': float(m.x),
                'y': float(m.y),
                'angle': float(m.angle),
                'kind': m.kind,
                'quality': float(m.quality)
            }
            for m in (template.minutiae or [])
        ],
        'fused': bool(template.fused),
        'source_count': int(template.source_count),
        'consensus_score': float(template.consensus_score)
    }


def template_from_secure_dict(data: Dict[str, Any]) -> FingerprintTemplate:
    """Deserialize template from secure dictionary.
    
    Reconstructs a FingerprintTemplate with:
    - Protected hash (restored from bytes)
    - Minutiae (restored from list of dicts)
    - raw_features = None (NEVER stored, can be recomputed if needed)
    
    Args:
        data: Dictionary with secure template data
        
    Returns:
        FingerprintTemplate object (with raw_features=None)
    """
    return FingerprintTemplate(
        identifier=data['identifier'],
        image_path=Path(data['image_path']),
        protected=np.frombuffer(data['protected'], dtype=np.uint8),
        bit_length=data['bit_length'],
        quality=data['quality'],
        raw_features=None,  # SECURITY: Never persist raw features
        minutiae=[
            Minutia(
                x=m['x'],
                y=m['y'],
                angle=m['angle'],
                kind=m['kind'],
                quality=m['quality']
            )
            for m in data.get('minutiae', [])
        ],
        fused=data.get('fused', False),
        source_count=data.get('source_count', 1),
        consensus_score=data.get('consensus_score', 1.0)
    )


def template_to_json(template: FingerprintTemplate) -> str:
    """Serialize template to JSON string (Protected as base64).
    
    Useful for REST APIs and portable storage.
    
    Args:
        template: FingerprintTemplate object
        
    Returns:
        JSON string
    """
    import base64
    
    data = template_to_secure_dict(template)
    # Convert bytes to base64 for JSON compatibility
    data['protected'] = base64.b64encode(data['protected']).decode('ascii')
    
    return json.dumps(data, indent=2)


def template_from_json(json_str: str) -> FingerprintTemplate:
    """Deserialize template from JSON string.
    
    Args:
        json_str: JSON string with template data
        
    Returns:
        FingerprintTemplate object
    """
    import base64
    
    data = json.loads(json_str)
    # Convert base64 back to bytes
    data['protected'] = base64.b64decode(data['protected'])
    
    return template_from_secure_dict(data)


# Backward compatibility aliases (for gradual migration)
serialize_template = template_to_secure_dict
deserialize_template = template_from_secure_dict
