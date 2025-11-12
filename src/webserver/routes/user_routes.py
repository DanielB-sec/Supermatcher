"""
User Management Routes
CRUD operations for fingerprint users.
"""

from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException
from typing import List
import asyncio
import tempfile
import os
import pickle

from ..database import BiometricDatabase
from ..auth import require_auth, rate_limit
from ..logger import log_biometric
from ..config import PRIVILEGE_ADMIN, QUALITY_THRESHOLD


router = APIRouter(tags=["Users"])


# Global reference to process pool and cache (set by server.py)
process_pool = None
template_cache = {}


def set_globals(pool, cache):
    """Set global process pool and template cache references."""
    global process_pool, template_cache
    process_pool = pool
    template_cache = cache


@router.post("", dependencies=[Depends(require_auth(PRIVILEGE_ADMIN)), Depends(rate_limit("enroll"))])
async def add_user(
    user_id: str = Form(...),
    name: str = Form(...),
    images: List[UploadFile] = File(...)
):
    """
    Enroll a new user with fingerprint images.
    
    Form Data:
        - user_id: User identifier
        - name: User name
        - images: List of fingerprint images (PNG, JPG, TIF, etc.)
    
    Returns:
        - user_id: Enrolled user ID
        - quality: Template quality score
        - num_images: Number of images processed
    """
    from ..biometric_worker import worker_enroll
    
    # DEBUG: Log received parameters
    print(f"[DEBUG add_user] Received user_id='{user_id}', name='{name}', num_images={len(images)}")
    
    # Save uploaded files to temp
    temp_files = []
    try:
        for img in images:
            content = await img.read()
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            temp.write(content)
            temp.close()
            temp_files.append(temp.name)
        
        # Prepare settings for supermatcher v1.0
        settings = {
            'quality_threshold': QUALITY_THRESHOLD,
            'fusion': {
                'enabled': True,
                'distance': 12.0,
                'angle_deg': 15.0,
                'min_consensus': 0.5,
                'keep_raw': False,
                'mode': 'optimal'
            }
        }
        
        # Run enrollment in ProcessPool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            process_pool,
            worker_enroll,
            temp_files,
            user_id,
            settings
        )
        
        if not result['success']:
            raise HTTPException(
                status_code=400,
                detail=f"Enrollment failed: {result['error']}"
            )
        
        # Deserialize template
        from src.models_serialization import template_from_secure_dict
        template = template_from_secure_dict(result['template_secure'])
        
        # Save to database
        db = BiometricDatabase()
        success = db.add_fingerprint(
            user_id=user_id,
            name=name,
            template_obj=template
        )
        
        if not success:
            raise HTTPException(
                status_code=409,
                detail=f"User already exists: {user_id}"
            )
        
        # Update cache
        template_cache[user_id] = template
        
        # Log
        log_biometric(
            "ENROLL",
            user_id,
            "SUCCESS",
            details={'quality': result['quality'], 'num_images': result['num_images']},
            performed_by="admin"
        )
        
        return {
            "user_id": user_id,
            "quality": result['quality'],
            "num_images": result['num_images']
        }
    
    finally:
        # Cleanup temp files
        for f in temp_files:
            try:
                os.unlink(f)
            except:
                pass


@router.get("", dependencies=[Depends(require_auth())])
async def list_users():
    """
    List all enrolled users.
    
    Returns:
        - users: List of user objects (without template data)
    """
    db = BiometricDatabase()
    users = db.list_fingerprints()
    
    return {"users": users}


@router.get("/{user_id}", dependencies=[Depends(require_auth())])
async def get_user(user_id: str):
    """
    Get user details by ID.
    
    Path Parameters:
        - user_id: User identifier
    
    Returns:
        User object (without template data)
    """
    db = BiometricDatabase()
    user = db.get_fingerprint(user_id)
    
    if not user:
        raise HTTPException(
            status_code=404,
            detail=f"User not found: {user_id}"
        )
    
    # Return without template data (too large)
    return {
        "user_id": user["user_id"],
        "name": user.get("name", ""),
        "quality": user.get("quality", 0.0),
        "created_at": user.get("created_at", ""),
        "updated_at": user.get("updated_at", ""),
        "created_by": user.get("created_by", "")
    }


@router.put("/{user_id}", dependencies=[Depends(require_auth(PRIVILEGE_ADMIN)), Depends(rate_limit("enroll"))])
async def update_user(
    user_id: str,
    name: str = Form(None),
    images: List[UploadFile] = File(None)
):
    """
    Update user details and/or re-enroll with new images.
    
    Path Parameters:
        - user_id: User identifier
    
    Form Data (all optional):
        - name: New user name
        - images: New fingerprint images (triggers re-enrollment)
    
    Returns:
        - message: Success message
        - user_id: Updated user ID
    """
    from ..biometric_worker import worker_enroll
    
    db = BiometricDatabase()
    
    # Check if user exists
    existing = db.get_fingerprint(user_id)
    if not existing:
        raise HTTPException(
            status_code=404,
            detail=f"User not found: {user_id}"
        )
    
    template = None
    
    # Re-enroll if images provided
    if images:
        temp_files = []
        try:
            for img in images:
                content = await img.read()
                temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                temp.write(content)
                temp.close()
                temp_files.append(temp.name)
            
            settings = {
                'quality_threshold': QUALITY_THRESHOLD,
                'fusion': {
                    'enabled': True,
                    'distance': 12.0,
                    'angle_deg': 15.0,
                    'min_consensus': 0.5,
                    'keep_raw': False,
                    'mode': 'optimal'
                }
            }
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                process_pool,
                worker_enroll,
                temp_files,
                user_id,
                settings
            )
            
            if not result['success']:
                raise HTTPException(
                    status_code=400,
                    detail=f"Re-enrollment failed: {result['error']}"
                )
            
            from src.models_serialization import template_from_secure_dict
            template = template_from_secure_dict(result['template_secure'])
        
        finally:
            for f in temp_files:
                try:
                    os.unlink(f)
                except:
                    pass
    
    # Update user
    success = db.update_fingerprint(user_id, name, template)
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to update user"
        )
    
    # Update cache if template changed
    if template:
        template_cache[user_id] = template
    
    log_biometric("UPDATE", user_id, "SUCCESS", performed_by="admin")
    
    return {
        "message": "User updated successfully",
        "user_id": user_id
    }


@router.delete("/{user_id}", dependencies=[Depends(require_auth(PRIVILEGE_ADMIN))])
async def delete_user(user_id: str):
    """
    Delete a user.
    
    Path Parameters:
        - user_id: User identifier
    
    Returns:
        - message: Success message
        - user_id: Deleted user ID
    """
    db = BiometricDatabase()
    success = db.delete_fingerprint(user_id)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"User not found: {user_id}"
        )
    
    # Remove from cache
    template_cache.pop(user_id, None)
    
    log_biometric("DELETE", user_id, "SUCCESS", performed_by="admin")
    
    return {
        "message": "User deleted successfully",
        "user_id": user_id
    }


@router.delete("", dependencies=[Depends(require_auth(PRIVILEGE_ADMIN))])
async def delete_all_users():
    """
    Delete all users (admin only).
    
    Returns:
        - message: Success message
        - count: Number of users deleted
    """
    db = BiometricDatabase()
    count = db.delete_all_fingerprints()
    
    # Clear cache
    template_cache.clear()
    
    log_biometric("DELETE_ALL", None, "SUCCESS", details={'count': count}, performed_by="admin")
    
    return {
        "message": f"Deleted {count} users",
        "count": count
    }
