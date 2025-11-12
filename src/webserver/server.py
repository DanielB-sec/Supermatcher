"""
FastAPI WebServer - Main Application
Production-ready biometric authentication server with JWT, rate limiting, and ProcessPool.
"""

import asyncio
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    HOST, PORT_HTTPS, PORT_HTTP,
    MAX_WORKERS, USE_CPU_AFFINITY, PCORES,
    FRONTEND_PATH, VERBOSE
)
from .database import BiometricDatabase
from .jobs import JobManager
from .logger import log_startup, log_shutdown, log_access, get_logger
from .auth import get_client_ip, cleanup_rate_limit_storage

# Import routes
from .routes import auth_router, user_router, biometric_router, admin_router
from .routes import user_routes, biometric_routes, admin_routes


# Create FastAPI app
app = FastAPI(
    title="Biometric WebServer",
    version="1.0.0",
    description="Production-ready fingerprint authentication system"
)


# Global resources
process_pool = None
template_cache = {}
job_manager = None
db = None

logger = get_logger("server")


def init_worker_affinity():
    """
    Initializer function for ProcessPool workers.
    Sets CPU affinity to P-cores only (must be module-level for Windows pickle).
    """
    try:
        import psutil
        from .config import PCORES
        
        p = psutil.Process()
        p.cpu_affinity(PCORES)
        logger.info(f"✓ Worker {p.pid}: CPU affinity set to P-cores {PCORES}")
    except Exception as e:
        logger.warning(f"✗ Worker: Failed to set CPU affinity: {e}")


@app.on_event("startup")
async def startup():
    """Initialize server resources on startup."""
    global process_pool, template_cache, job_manager, db
    
    logger.info("Starting biometric webserver...")
    
    # Initialize database
    db = BiometricDatabase()
    logger.info("✓ Database initialized")
    
    # Initialize ProcessPool with CPU affinity initializer
    if USE_CPU_AFFINITY and PCORES:
        process_pool = ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            initializer=init_worker_affinity
        )
        logger.info(f"✓ ProcessPoolExecutor initialized ({MAX_WORKERS} workers with P-core affinity)")
    else:
        process_pool = ProcessPoolExecutor(max_workers=MAX_WORKERS)
        logger.info(f"✓ ProcessPoolExecutor initialized ({MAX_WORKERS} workers)")
    
    # Load all templates into cache
    templates = db.get_all_templates()
    template_cache.clear()
    for user_id, template in templates:
        template_cache[user_id] = template
    
    logger.info(f"✓ Cached {len(template_cache)} templates")
    
    # Initialize job manager
    job_manager = JobManager(db, process_pool)
    logger.info("✓ Job manager initialized")
    
    # Set global references in route modules
    user_routes.set_globals(process_pool, template_cache)
    biometric_routes.set_globals(process_pool, template_cache)
    admin_routes.set_job_manager(job_manager)
    
    # Log startup info
    log_startup({
        'host': HOST,
        'port_https': PORT_HTTPS,
        'port_http': PORT_HTTP,
        'templates_cached': len(template_cache),
        'max_workers': MAX_WORKERS,
        'cpu_affinity': USE_CPU_AFFINITY
    })
    
    logger.info("="*70)
    logger.info("BIOMETRIC WEBSERVER READY")
    logger.info("="*70)


@app.on_event("shutdown")
async def shutdown():
    """Cleanup resources on shutdown."""
    global process_pool, db
    
    logger.info("Shutting down biometric webserver...")
    
    log_shutdown()
    
    # Shutdown ProcessPool
    if process_pool:
        process_pool.shutdown(wait=True)
        logger.info("✓ ProcessPool shut down")
    
    # Close database
    if db:
        db.close()
        logger.info("✓ Database closed")
    
    logger.info("Shutdown complete")


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests."""
    start = time.time()
    
    # Get client IP
    ip = get_client_ip(request)
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = (time.time() - start) * 1000
    
    # Get user from request state (if authenticated)
    user = getattr(request.state, 'user', {}).get('login')
    
    # Log access
    log_access(
        ip=ip,
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code,
        duration_ms=duration,
        user=user
    )
    
    return response


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Static files (mount AFTER defining routes to avoid conflicts)
if FRONTEND_PATH.exists():
    # Mount static files for CSS/JS
    app.mount("/static", StaticFiles(directory=FRONTEND_PATH), name="static")
    
    @app.get("/")
    async def index():
        """Serve index.html (login page)."""
        index_file = FRONTEND_PATH / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        else:
            return {"message": "Biometric WebServer API", "version": "1.0.0"}
    
    @app.get("/admin")
    async def admin_page():
        """Serve admin dashboard."""
        admin_file = FRONTEND_PATH / "admin.html"
        if admin_file.exists():
            return FileResponse(admin_file)
        else:
            return {"error": "Admin page not found"}
    
    @app.get("/endpoint")
    async def endpoint_page():
        """Serve endpoint dashboard."""
        endpoint_file = FRONTEND_PATH / "endpoint.html"
        if endpoint_file.exists():
            return FileResponse(endpoint_file)
        else:
            return {"error": "Endpoint page not found"}
else:
    logger.warning(f"Frontend directory not found: {FRONTEND_PATH}")


# Include routers
app.include_router(auth_router, prefix="/api", tags=["Authentication"])
app.include_router(user_router, prefix="/api/users", tags=["Users"])
app.include_router(biometric_router, prefix="/api", tags=["Biometric"])
app.include_router(admin_router, prefix="/api/admin", tags=["Admin"])


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "cached_templates": len(template_cache),
        "max_workers": MAX_WORKERS,
        "running_jobs": len(job_manager.running_jobs) if job_manager else 0
    }


# Periodic cleanup task
async def periodic_cleanup():
    """Periodic cleanup of rate limit storage."""
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        cleanup_rate_limit_storage()
        logger.info("Rate limit storage cleaned up")


# Start cleanup task on startup
@app.on_event("startup")
async def start_cleanup_task():
    """Start periodic cleanup task."""
    asyncio.create_task(periodic_cleanup())


# Export app
__all__ = ['app']


if __name__ == "__main__":
    import uvicorn
    
    # Run with uvicorn
    uvicorn.run(
        "src.webserver.server:app",
        host=HOST,
        port=PORT_HTTP,
        reload=False,
        log_level="info" if VERBOSE else "warning"
    )
