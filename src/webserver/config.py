"""WebServer Configuration

Centralized configuration for the biometric webserver.
All settings can be adjusted here without modifying the source code.

"""

import os
from pathlib import Path
from typing import List, Tuple

# ============================================================================
# PATHS
# ============================================================================

# Project root and paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
WEBSERVER_ROOT = Path(__file__).parent
FRONTEND_PATH = WEBSERVER_ROOT / "frontend"

# Database
DB_PATH = PROJECT_ROOT / "biometric_system.db"
DB_ENCRYPTION_KEY = None  # Auto-generated on first run, stored in .db_key file
DB_KEY_FILE = PROJECT_ROOT / ".db_key"

# SSL Certificates
SSL_CERT_DIR = PROJECT_ROOT / "certs"
SSL_CERT_FILE = SSL_CERT_DIR / "cert.pem"
SSL_KEY_FILE = SSL_CERT_DIR / "key.pem"

# Logs
LOG_DIR = PROJECT_ROOT / "logs"
LOG_ACCESS = LOG_DIR / "access.log"
LOG_AUTH = LOG_DIR / "auth.log"
LOG_BIOMETRIC = LOG_DIR / "biometric.log"
LOG_ERROR = LOG_DIR / "error.log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB per file
LOG_BACKUP_COUNT = 5  # Keep 5 backup files

# ============================================================================
# SERVER CONFIGURATION
# ============================================================================

# Network
HOST = "0.0.0.0"
PORT_HTTPS = 8443
PORT_HTTP = 8080  # Fallback if SSL fails

# CORS
CORS_ORIGINS = ["*"]  # Allow all origins (adjust for production)
CORS_ALLOW_CREDENTIALS = True

# Request limits
MAX_REQUEST_SIZE = 100 * 1024 * 1024  # 100MB (for large fingerprint batches)
REQUEST_TIMEOUT = 300  # 5 minutes for long operations

# ============================================================================
# AUTHENTICATION & SECURITY
# ============================================================================

# JWT
JWT_SECRET_KEY = None  # Auto-generated on first run, stored in .jwt_secret file
JWT_SECRET_FILE = PROJECT_ROOT / ".jwt_secret"
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_ADMIN_HOURS = 1  # Admin tokens expire in 1 hour
JWT_EXPIRY_ENDPOINT_HOURS = 12  # Endpoint tokens expire in 12 hours

# Password requirements
PASSWORD_MIN_LENGTH = 6
PASSWORD_REQUIRE_UPPERCASE = False
PASSWORD_REQUIRE_LOWERCASE = False
PASSWORD_REQUIRE_DIGIT = False
PASSWORD_REQUIRE_SPECIAL = False

# Privileges
PRIVILEGE_ADMIN = "admin"
PRIVILEGE_ENDPOINT = "endpoint"

# Rate Limiting (requests, seconds)
# Format: (max_requests, time_window_seconds)
RATE_LIMITS = {
    "login": (5, 300),          # 5 attempts per 5 minutes
    "verify": (10, 60),         # 10 verifications per minute
    "identify": (5, 60),        # 5 identifications per minute
    "enroll": (20, 3600),       # 20 enrollments per hour
    "update": (10, 3600),       # 10 updates per hour
    "delete": (30, 3600),       # 30 deletes per hour
    "load_folder": (2, 3600),   # 2 folder loads per hour
    "list_users": (60, 60),     # 60 lists per minute
}

# Account lockout
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 15

# ============================================================================
# BIOMETRIC CONFIGURATION
# ============================================================================

# Thresholds (adjusted based on benchmark_production_v1_0)
QUALITY_THRESHOLD = 0.30  # Minimum quality for enrollment
VERIFICATION_THRESHOLD = 0.75  # 1:1 matching threshold (combined hash+geometric score)
IDENTIFICATION_THRESHOLD = 0.70  # 1:N matching threshold (combined hash+geometric score)

# Enrollment
MIN_SAMPLES_PER_USER = 2  # Minimum images for enrollment
MAX_SAMPLES_PER_USER = 10  # Maximum images for enrollment
RECOMMENDED_SAMPLES = 5  # Recommended number of samples

# Fusion settings (from supermatcher v1.0 FusionSettings)
FUSION_ENABLED = True
FUSION_DISTANCE = 12.0
FUSION_ANGLE_DEG = 15.0
FUSION_MIN_CONSENSUS = 0.5
FUSION_KEEP_RAW = False
FUSION_MODE = "optimal"

# ============================================================================
# MULTIPROCESSING
# ============================================================================

# ProcessPool configuration
MAX_WORKERS = 4  # Number of parallel biometric operations
USE_CPU_AFFINITY = True
# Intel hybrid CPU: 6 P-cores (0-5) with hyperthreading (0-11) + 8 E-cores (12-19)
# Using only P-cores for maximum performance
PCORES = list(range(0, 12))  # P-cores logical threads (6 physical P-cores)

# Worker timeout
ENROLL_TIMEOUT = 120  # 2 minutes per enrollment
VERIFY_TIMEOUT = 30  # 30 seconds per verification
IDENTIFY_TIMEOUT = 60  # 1 minute per identification
LOAD_FOLDER_TIMEOUT = 3600  # 1 hour for folder loading

# ============================================================================
# BACKGROUND JOBS
# ============================================================================

# Job cleanup
JOB_RETENTION_HOURS = 24  # Keep completed jobs for 24 hours
JOB_CHECK_INTERVAL = 60  # Check for stale jobs every minute

# ============================================================================
# CACHE CONFIGURATION
# ============================================================================

# Template cache
CACHE_ENABLED = True
CACHE_RELOAD_ON_STARTUP = True  # Load all templates into RAM on startup

# ============================================================================
# DEFAULT USERS
# ============================================================================

# Default credentials (only used if DB doesn't exist)
DEFAULT_ADMIN_USER = "admin"
DEFAULT_ADMIN_PASS = "admin123"
DEFAULT_ENDPOINT_USER = "endpoint"
DEFAULT_ENDPOINT_PASS = "endpoint123"

# Verbose output
VERBOSE = True  # Set to False to reduce console output


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_db_key() -> str:
    """Get or generate database encryption key."""
    if DB_KEY_FILE.exists():
        return DB_KEY_FILE.read_text().strip()
    
    # Generate new key
    import secrets
    key = secrets.token_urlsafe(32)
    
    # Save key
    DB_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    DB_KEY_FILE.write_text(key)
    DB_KEY_FILE.chmod(0o600)  # Read/write for owner only
    
    return key


def get_jwt_secret() -> str:
    """Get or generate JWT secret key."""
    if JWT_SECRET_FILE.exists():
        return JWT_SECRET_FILE.read_text().strip()
    
    # Generate new secret
    import secrets
    secret = secrets.token_urlsafe(64)
    
    # Save secret
    JWT_SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
    JWT_SECRET_FILE.write_text(secret)
    JWT_SECRET_FILE.chmod(0o600)  # Read/write for owner only
    
    return secret


def ensure_directories():
    """Ensure all required directories exist."""
    dirs = [
        SSL_CERT_DIR,
        LOG_DIR,
        DB_PATH.parent,
    ]
    
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


# Auto-initialize on import
if DB_ENCRYPTION_KEY is None:
    DB_ENCRYPTION_KEY = get_db_key()

if JWT_SECRET_KEY is None:
    JWT_SECRET_KEY = get_jwt_secret()

ensure_directories()
