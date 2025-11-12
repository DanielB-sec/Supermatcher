"""
Admin Routes
System administration endpoints (stats, jobs, folder loading).
"""

from fastapi import APIRouter, Depends, HTTPException, Form
from pathlib import Path

from ..database import BiometricDatabase
from ..auth import require_auth
from ..logger import log_job
from ..config import PRIVILEGE_ADMIN


router = APIRouter(tags=["Admin"])


# Global reference to job manager (set by server.py)
job_manager = None


def set_job_manager(jm):
    """Set global job manager reference."""
    global job_manager
    job_manager = jm


@router.post("/load_folder", dependencies=[Depends(require_auth(PRIVILEGE_ADMIN))])
async def load_folder(
    folder_path: str = Form(...)
):
    """
    Start a background job to load fingerprints from folder.
    
    Expected folder structure:
        folder_path/
            user1/
                img1.png
                img2.png
            user2/
                img1.png
                img2.png
    
    Form Data:
        - folder_path: Path to folder containing user subfolders
    
    Returns:
        - job_id: Background job ID (UUID)
        - message: Info message
    """
    if not Path(folder_path).exists():
        raise HTTPException(
            status_code=400,
            detail=f"Folder not found: {folder_path}"
        )
    
    # Create background job
    job_id = await job_manager.create_load_folder_job(
        folder_path=folder_path,
        created_by="admin"  # TODO: Get from request context
    )
    
    return {
        "job_id": job_id,
        "message": f"Background job started to load folder: {folder_path}"
    }


@router.get("/jobs/{job_id}", dependencies=[Depends(require_auth(PRIVILEGE_ADMIN))])
async def get_job(job_id: str):
    """
    Get job status and results.
    
    Path Parameters:
        - job_id: Job ID (UUID)
    
    Returns:
        Job object with status, result, error, etc.
    """
    job = job_manager.get_job_status(job_id)
    
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}"
        )
    
    return job


@router.get("/jobs", dependencies=[Depends(require_auth(PRIVILEGE_ADMIN))])
async def list_running_jobs():
    """
    List all currently running jobs.
    
    Returns:
        - jobs: Dict mapping job_id -> job_info
    """
    jobs = job_manager.get_running_jobs()
    
    return {"jobs": jobs}


@router.delete("/jobs/{job_id}", dependencies=[Depends(require_auth(PRIVILEGE_ADMIN))])
async def cancel_job(job_id: str):
    """
    Cancel a running job.
    
    Path Parameters:
        - job_id: Job ID (UUID)
    
    Returns:
        - message: Success/failure message
        - cancelled: True if cancelled, False if not running
    """
    cancelled = job_manager.cancel_job(job_id)
    
    if cancelled:
        return {
            "message": f"Job cancelled: {job_id}",
            "cancelled": True
        }
    else:
        return {
            "message": f"Job not running or already completed: {job_id}",
            "cancelled": False
        }


@router.get("/stats", dependencies=[Depends(require_auth(PRIVILEGE_ADMIN))])
async def get_stats():
    """
    Get system statistics.
    
    Returns:
        - num_users: Number of enrolled users
        - avg_quality: Average template quality
        - encrypted: Whether database is encrypted
        - cached_templates: Number of templates in cache
        - running_jobs: Number of running background jobs
    """
    db = BiometricDatabase()
    stats = db.get_stats()
    
    # Add cache and job stats
    from ..routes.user_routes import template_cache
    
    stats['cached_templates'] = len(template_cache)
    stats['running_jobs'] = len(job_manager.running_jobs)
    
    return stats


@router.get("/audit", dependencies=[Depends(require_auth(PRIVILEGE_ADMIN))])
async def get_audit_log(
    limit: int = 100
):
    """
    Get audit log entries.
    
    Query Parameters:
        - limit: Maximum number of entries (default 100)
    
    Returns:
        - logs: List of audit log entries
    """
    db = BiometricDatabase()
    logs = db.get_audit_log(limit=limit)
    
    return {"logs": logs}


@router.post("/cleanup_jobs", dependencies=[Depends(require_auth(PRIVILEGE_ADMIN))])
async def cleanup_jobs(
    days: int = Form(7)
):
    """
    Clean up old completed/failed jobs.
    
    Form Data:
        - days: Age threshold in days (default 7)
    
    Returns:
        - message: Success message
        - removed: Number of jobs removed
    """
    count = job_manager.cleanup_old_jobs(days)
    
    return {
        "message": f"Cleaned up {count} old jobs",
        "removed": count
    }
