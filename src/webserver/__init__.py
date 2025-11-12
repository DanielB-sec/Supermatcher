"""
Webserver Package - Biometric Authentication System
FastAPI-based REST API with JWT authentication and ProcessPool biometric operations.
"""

from .server import app
from .database import BiometricDatabase
from .auth import create_token, verify_token, require_auth
from .logger import get_logger

__version__ = "1.0.0"
__all__ = ['app', 'BiometricDatabase', 'create_token', 'verify_token', 'require_auth', 'get_logger']
