"""
Biometric Operations Routes
Verify (1:1) and Identify (1:N) operations.
"""

from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException
import asyncio
import tempfile
import os
import pickle

from ..database import BiometricDatabase
from ..auth import require_auth, rate_limit
from ..logger import log_biometric
from ..config import VERIFICATION_THRESHOLD, IDENTIFICATION_THRESHOLD


router = APIRouter(tags=["Biometric Operations"])


# Global reference to process pool and cache (set by server.py)
process_pool = None
template_cache = {}


def set_globals(pool, cache):
    """Set global process pool and template cache references."""
    global process_pool, template_cache
    process_pool = pool
    template_cache = cache


@router.post("/verify", dependencies=[Depends(require_auth()), Depends(rate_limit("verify"))])
async def verify(
    user_id: str = Form(...),
    image: UploadFile = File(...)
):
    """
    1:1 verification against enrolled user.
    
    Form Data:
        - user_id: User to verify against
        - image: Probe fingerprint image
    
    Returns:
        - match: True if verified, False otherwise
        - score: Matching score
        - user_id: Claimed user ID
        - threshold: Verification threshold used
    """
    from ..biometric_worker import worker_verify
    from src.models_serialization import template_to_secure_dict
    
    # Get template from cache
    template = template_cache.get(user_id)
    if not template:
        raise HTTPException(
            status_code=404,
            detail=f"User not found: {user_id}"
        )
    
    # Save probe image to temp file
    content = await image.read()
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp.write(content)
    temp.close()
    
    try:
        # Run verification in ProcessPool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            process_pool,
            worker_verify,
            temp.name,
            template_to_secure_dict(template),
            VERIFICATION_THRESHOLD,
            {}  # Settings dict (unused for now)
        )
        
        if not result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Verification failed: {result['error']}"
            )
        
        # Log
        log_biometric(
            "VERIFY",
            user_id,
            "MATCH" if result['match'] else "NO_MATCH",
            details={
                'score': result['score'],
                'hash_score': result.get('hash_score', 0),
                'geometric_score': result.get('geometric_score', 0),
                'num_matched': result['num_matched']
            },
            performed_by="user"
        )
        
        return {
            "match": result['match'],
            "score": result['score'],
            "hash_score": result.get('hash_score', result['score']),
            "geometric_score": result.get('geometric_score', 0),
            "user_id": user_id,
            "threshold": VERIFICATION_THRESHOLD,
            "num_matched": result['num_matched']
        }
    
    finally:
        # Cleanup temp file
        try:
            os.unlink(temp.name)
        except:
            pass


@router.post("/identify", dependencies=[Depends(require_auth()), Depends(rate_limit("identify"))])
async def identify(
    image: UploadFile = File(...),
    top_k: int = Form(5)
):
    """
    1:N identification against all enrolled users.
    
    Form Data:
        - image: Probe fingerprint image
        - top_k: Number of top matches to return (default 5)
    
    Returns:
        - identified: True if match found above threshold
        - best_match_id: User ID of best match (if identified)
        - best_score: Score of best match
        - matches: List of top matches [{user_id, score, num_matched}]
        - total_users: Total number of enrolled users
        - threshold: Identification threshold used
    """
    from ..biometric_worker import worker_identify
    
    if not template_cache:
        raise HTTPException(
            status_code=400,
            detail="No users enrolled in database"
        )
    
    # Save probe image to temp file
    content = await image.read()
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp.write(content)
    temp.close()
    
    try:
        # Serialize all templates for ProcessPool (secure dicts)
        from src.models_serialization import template_to_secure_dict
        gallery = {uid: template_to_secure_dict(tpl) for uid, tpl in template_cache.items()}
        
        # Run identification in ProcessPool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            process_pool,
            worker_identify,
            temp.name,
            gallery,
            IDENTIFICATION_THRESHOLD,
            {},  # Settings dict (unused for now)
            top_k
        )
        
        if not result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Identification failed: {result['error']}"
            )
        
        # Log
        log_biometric(
            "IDENTIFY",
            result['best_match_id'] if result['identified'] else None,
            "MATCH" if result['identified'] else "NO_MATCH",
            details={
                'best_score': result['best_score'],
                'num_candidates': len(result['matches'])
            },
            performed_by="user"
        )
        
        return {
            "identified": result['identified'],
            "best_match_id": result['best_match_id'],
            "best_score": result['best_score'],
            "matches": result['matches'],
            "total_users": len(template_cache),
            "threshold": IDENTIFICATION_THRESHOLD
        }
    
    finally:
        # Cleanup temp file
        try:
            os.unlink(temp.name)
        except:
            pass
