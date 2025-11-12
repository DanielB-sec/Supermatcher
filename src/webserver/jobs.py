"""
Background Job Management
Manages long-running tasks (folder loading, batch operations) with database persistence.
"""

import asyncio
import uuid
import json
from typing import Dict, Optional, Any
from concurrent.futures import ProcessPoolExecutor, Future
from datetime import datetime

from .database import BiometricDatabase
from .biometric_worker import worker_load_folder
from .logger import log_job, log_error
from .config import QUALITY_THRESHOLD


class JobManager:
    """Manages background jobs with database persistence and progress tracking."""
    
    def __init__(self, db: BiometricDatabase, executor: ProcessPoolExecutor):
        """
        Initialize job manager.
        
        Args:
            db: Database instance
            executor: ProcessPoolExecutor for running jobs
        """
        self.db = db
        self.executor = executor
        self.running_jobs: Dict[str, Future] = {}
    
    async def create_load_folder_job(
        self,
        folder_path: str,
        created_by: str
    ) -> str:
        """
        Create and start a folder loading job.
        
        Args:
            folder_path: Path to folder with user subfolders
            created_by: Username who created the job
        
        Returns:
            Job ID (UUID)
        """
        job_id = str(uuid.uuid4())
        
        # Create job in database
        self.db.create_job(
            job_id=job_id,
            job_type='load_folder',
            created_by=created_by
        )
        
        log_job(job_id, 'LOAD_FOLDER', 'CREATED', created_by=created_by)
        
        # Prepare settings
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
        
        # Submit to ProcessPool
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            self.executor,
            worker_load_folder,
            folder_path,
            settings,
            None  # No progress callback (could add later)
        )
        
        self.running_jobs[job_id] = future
        
        # Update status to processing
        self.db.update_job(job_id, status='processing')
        log_job(job_id, 'LOAD_FOLDER', 'PROCESSING')
        
        # Track job completion asynchronously
        asyncio.create_task(self._track_load_folder_job(job_id, future, created_by))
        
        return job_id
    
    async def _track_load_folder_job(
        self,
        job_id: str,
        future: Future,
        created_by: str
    ):
        """
        Track job completion and save enrolled users to database.
        
        Args:
            job_id: Job ID
            future: Future object from ProcessPool
            created_by: Username who created the job
        """
        try:
            # Wait for completion
            result = await asyncio.wrap_future(future)
            
            if not result['success']:
                # Job failed
                self.db.update_job(
                    job_id=job_id,
                    status='failed',
                    error=result.get('error', 'Unknown error')
                )
                log_job(job_id, 'LOAD_FOLDER', 'FAILED', details={'error': result.get('error')})
                return
            
            # Job succeeded - save enrolled users to database
            enrolled_count = 0
            failed_count = len(result['failed'])
            
            import pickle
            
            # Import template_cache to update it
            from .routes.user_routes import template_cache
            
            for enrollment in result['enrolled']:
                user_id = enrollment['user_id']
                template_secure_dict = enrollment['template_secure']
                
                # Deserialize template
                from src.models_serialization import template_from_secure_dict
                template = template_from_secure_dict(template_secure_dict)
                
                # Add to database
                success = self.db.add_fingerprint(
                    user_id=user_id,
                    name=user_id,  # Use user_id as name (can be updated later)
                    template_obj=template,
                    username=created_by
                )
                
                if success:
                    enrolled_count += 1
                    # IMPORTANT: Update template cache
                    template_cache[user_id] = template
                    print(f"[DEBUG] Added user {user_id} to cache (total: {len(template_cache)})")
            
            # Update job status
            result_summary = {
                'enrolled': enrolled_count,
                'failed': failed_count,
                'total': result['total'],
                'failed_users': result['failed']
            }
            
            self.db.update_job(
                job_id=job_id,
                status='completed',
                result=json.dumps(result_summary)
            )
            
            log_job(
                job_id,
                'LOAD_FOLDER',
                'COMPLETED',
                details={'enrolled': enrolled_count, 'failed': failed_count},
                created_by=created_by
            )
        
        except Exception as e:
            # Unexpected error
            self.db.update_job(
                job_id=job_id,
                status='failed',
                error=str(e)
            )
            log_error(e, context=f"track_job:{job_id}", user=created_by)
            log_job(job_id, 'LOAD_FOLDER', 'FAILED', details={'error': str(e)})
        
        finally:
            # Remove from running jobs
            self.running_jobs.pop(job_id, None)
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job status from database.
        
        Args:
            job_id: Job ID
        
        Returns:
            Job dict or None if not found
        """
        return self.db.get_job(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.
        
        Args:
            job_id: Job ID
        
        Returns:
            True if cancelled, False if not running
        """
        if job_id in self.running_jobs:
            future = self.running_jobs[job_id]
            cancelled = future.cancel()
            
            if cancelled:
                self.db.update_job(job_id, status='cancelled')
                log_job(job_id, 'UNKNOWN', 'CANCELLED')
                del self.running_jobs[job_id]
                return True
        
        return False
    
    def get_running_jobs(self) -> Dict[str, Any]:
        """
        Get list of currently running jobs.
        
        Returns:
            Dict mapping job_id -> job_info
        """
        running = {}
        
        for job_id in list(self.running_jobs.keys()):
            job_info = self.db.get_job(job_id)
            if job_info:
                running[job_id] = job_info
        
        return running
    
    def cleanup_old_jobs(self, days: int = 7):
        """
        Clean up old completed/failed jobs from database.
        
        Args:
            days: Age threshold in days
        """
        count = self.db.cleanup_old_jobs(days)
        log_job('cleanup', 'CLEANUP', 'COMPLETED', details={'removed': count})
        return count


# Test
if __name__ == "__main__":
    print("Testing job manager...")
    
    # Note: This requires a database and executor instance
    # For real testing, initialize properly
    
    print("\n✓ JobManager class defined")
    print("  - create_load_folder_job()")
    print("  - get_job_status()")
    print("  - cancel_job()")
    print("  - get_running_jobs()")
    print("  - cleanup_old_jobs()")
