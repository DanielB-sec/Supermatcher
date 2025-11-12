"""
Webserver Logging System
Provides structured logging to separate files with automatic rotation.

Log Files:
- access.log: HTTP requests (IP, endpoint, status, duration)
- auth.log: Authentication events (login, logout, failed attempts)
- biometric.log: Biometric operations (enroll, verify, identify results)
- error.log: Application errors and exceptions
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json

from .config import LOG_DIR, VERBOSE


# Ensure log directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)


# Log file paths
ACCESS_LOG = LOG_DIR / "access.log"
AUTH_LOG = LOG_DIR / "auth.log"
BIOMETRIC_LOG = LOG_DIR / "biometric.log"
ERROR_LOG = LOG_DIR / "error.log"


# Log formats
DETAILED_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
JSON_FORMAT = '%(message)s'  # For structured JSON logs


def _create_rotating_handler(
    log_file: Path,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    formatter_string: Optional[str] = None
) -> RotatingFileHandler:
    """
    Create a rotating file handler.
    
    Args:
        log_file: Path to the log file
        max_bytes: Max file size before rotation (default 10MB)
        backup_count: Number of backup files to keep (default 5)
        formatter_string: Log format string (default DETAILED_FORMAT)
    
    Returns:
        Configured RotatingFileHandler
    """
    handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    
    formatter = logging.Formatter(formatter_string or DETAILED_FORMAT)
    handler.setFormatter(formatter)
    
    return handler


def _get_logger(name: str, log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """
    Get or create a logger with rotating file handler.
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level (default INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    logger.propagate = False  # Don't propagate to root logger
    
    # Add rotating file handler
    handler = _create_rotating_handler(log_file)
    logger.addHandler(handler)
    
    # Add console handler if verbose
    if VERBOSE:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(DETAILED_FORMAT))
        logger.addHandler(console)
    
    return logger


# Create specialized loggers
access_logger = _get_logger("webserver.access", ACCESS_LOG)
auth_logger = _get_logger("webserver.auth", AUTH_LOG)
biometric_logger = _get_logger("webserver.biometric", BIOMETRIC_LOG)
error_logger = _get_logger("webserver.error", ERROR_LOG, level=logging.ERROR)


# Convenience functions

def log_access(
    ip: str,
    method: str,
    endpoint: str,
    status_code: int,
    duration_ms: float,
    user: Optional[str] = None
):
    """
    Log HTTP access.
    
    Args:
        ip: Client IP address
        method: HTTP method (GET, POST, etc.)
        endpoint: Request endpoint
        status_code: HTTP status code
        duration_ms: Request duration in milliseconds
        user: Authenticated username (optional)
    """
    user_info = f" user={user}" if user else ""
    access_logger.info(
        f"{ip} - {method} {endpoint} - {status_code} - {duration_ms:.2f}ms{user_info}"
    )


def log_auth(
    event: str,
    login: str,
    ip: str,
    success: bool = True,
    details: Optional[str] = None
):
    """
    Log authentication event.
    
    Args:
        event: Event type (LOGIN, LOGOUT, TOKEN_REFRESH, etc.)
        login: Username
        ip: Client IP address
        success: Whether the operation succeeded
        details: Additional details (optional)
    """
    status = "SUCCESS" if success else "FAILED"
    detail_info = f" - {details}" if details else ""
    
    auth_logger.info(f"{event} {status} - login={login} ip={ip}{detail_info}")


def log_biometric(
    operation: str,
    user_id: Optional[str],
    result: str,
    details: Optional[Dict[str, Any]] = None,
    performed_by: Optional[str] = None
):
    """
    Log biometric operation.
    
    Args:
        operation: Operation type (ENROLL, VERIFY, IDENTIFY)
        user_id: Target user ID (for enroll/verify) or None (for identify)
        result: Operation result (SUCCESS, FAILURE, NO_MATCH, etc.)
        details: Additional details dict (score, quality, matches, etc.)
        performed_by: Username who performed the operation
    """
    user_info = f"user_id={user_id}" if user_id else "user_id=None"
    by_info = f" by={performed_by}" if performed_by else ""
    
    # Format details
    detail_str = ""
    if details:
        detail_parts = [f"{k}={v}" for k, v in details.items()]
        detail_str = f" - {', '.join(detail_parts)}"
    
    biometric_logger.info(
        f"{operation} {result} - {user_info}{by_info}{detail_str}"
    )


def log_error(
    error: Exception,
    context: Optional[str] = None,
    user: Optional[str] = None,
    ip: Optional[str] = None
):
    """
    Log application error.
    
    Args:
        error: Exception object
        context: Context where error occurred (endpoint, function name, etc.)
        user: Authenticated username (optional)
        ip: Client IP address (optional)
    """
    context_info = f" in {context}" if context else ""
    user_info = f" user={user}" if user else ""
    ip_info = f" ip={ip}" if ip else ""
    
    error_logger.error(
        f"{type(error).__name__}: {str(error)}{context_info}{user_info}{ip_info}",
        exc_info=True  # Include stack trace
    )


def log_job(
    job_id: str,
    job_type: str,
    status: str,
    details: Optional[Dict[str, Any]] = None,
    created_by: Optional[str] = None
):
    """
    Log background job event.
    
    Args:
        job_id: Unique job ID
        job_type: Job type (LOAD_FOLDER, etc.)
        status: Job status (CREATED, PROCESSING, COMPLETED, FAILED)
        details: Additional details dict
        created_by: Username who created the job
    """
    by_info = f" by={created_by}" if created_by else ""
    
    detail_str = ""
    if details:
        detail_parts = [f"{k}={v}" for k, v in details.items()]
        detail_str = f" - {', '.join(detail_parts)}"
    
    access_logger.info(
        f"JOB {status} - type={job_type} id={job_id}{by_info}{detail_str}"
    )


def log_startup(info: Dict[str, Any]):
    """
    Log server startup information.
    
    Args:
        info: Startup info dict (host, port, ssl, num_templates, etc.)
    """
    access_logger.info("=" * 70)
    access_logger.info("WEBSERVER STARTING")
    access_logger.info("=" * 70)
    
    for key, value in info.items():
        access_logger.info(f"{key}: {value}")
    
    access_logger.info("=" * 70)


def log_shutdown():
    """Log server shutdown."""
    access_logger.info("=" * 70)
    access_logger.info("WEBSERVER SHUTTING DOWN")
    access_logger.info("=" * 70)


# JSON structured logging (for external log aggregation)

def log_json(
    logger_type: str,
    data: Dict[str, Any]
):
    """
    Log structured JSON data.
    
    Args:
        logger_type: Logger to use (access, auth, biometric, error)
        data: Dictionary to log as JSON
    """
    loggers = {
        'access': access_logger,
        'auth': auth_logger,
        'biometric': biometric_logger,
        'error': error_logger
    }
    
    logger = loggers.get(logger_type, access_logger)
    
    # Add timestamp
    data['timestamp'] = datetime.utcnow().isoformat()
    
    logger.info(json.dumps(data))


# Get logger by name (for custom usage)

def get_logger(name: str) -> logging.Logger:
    """
    Get a custom logger (writes to error.log).
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return _get_logger(f"webserver.{name}", ERROR_LOG)


# Cleanup old logs (optional utility)

def cleanup_old_logs(days: int = 30):
    """
    Delete log files older than specified days.
    
    Args:
        days: Age threshold in days
    """
    from datetime import timedelta
    import time
    
    cutoff = time.time() - (days * 86400)
    
    deleted = 0
    for log_file in LOG_DIR.glob("*.log*"):
        if log_file.stat().st_mtime < cutoff:
            log_file.unlink()
            deleted += 1
    
    access_logger.info(f"Cleaned up {deleted} old log files")


if __name__ == "__main__":
    # Test logging
    print("Testing webserver logging...")
    
    log_access("192.168.1.100", "POST", "/api/users", 201, 1234.56, user="admin")
    log_auth("LOGIN", "admin", "192.168.1.100", success=True)
    log_biometric("ENROLL", "user123", "SUCCESS", details={"quality": 0.85, "minutiae": 42})
    log_error(Exception("Test error"), context="/api/test", user="admin")
    
    print(f"\n✓ Logs written to {LOG_DIR}")
    print(f"  - {ACCESS_LOG}")
    print(f"  - {AUTH_LOG}")
    print(f"  - {BIOMETRIC_LOG}")
    print(f"  - {ERROR_LOG}")
