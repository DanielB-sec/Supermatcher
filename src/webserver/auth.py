"""
Webserver Authentication and Authorization
Provides JWT token management, rate limiting, and security decorators.

Features:
- JWT token creation/validation with configurable expiry
- Rate limiting per operation type (login, verify, identify, etc.)
- FastAPI dependency decorators for auth enforcement
- IP-based and user-based rate limiting
"""

import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from collections import defaultdict
from time import time
from functools import wraps

from fastapi import HTTPException, status, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .config import (
    JWT_SECRET_KEY,
    JWT_ALGORITHM,
    JWT_EXPIRY_ADMIN_HOURS,
    JWT_EXPIRY_ENDPOINT_HOURS,
    RATE_LIMITS,
    MAX_LOGIN_ATTEMPTS,
    PRIVILEGE_ADMIN,
    PRIVILEGE_ENDPOINT
)
from .logger import log_auth, log_error


# JWT Security
security = HTTPBearer()


# Rate limiting storage
# Format: {(identifier, operation): [(timestamp, ...), ...]}
_rate_limit_storage: Dict[Tuple[str, str], list] = defaultdict(list)

# Login attempt tracking
# Format: {ip: [(timestamp, failed), ...]}
_login_attempts: Dict[str, list] = defaultdict(list)


# ==================== JWT Token Management ====================

def create_token(user_data: Dict[str, str], privilege: Optional[str] = None) -> str:
    """
    Create a JWT token for authenticated user.
    
    Args:
        user_data: User information dict (must have 'login' and 'privilege' keys)
        privilege: Override privilege (optional, uses user_data['privilege'] by default)
    
    Returns:
        JWT token string
    
    Example:
        >>> token = create_token({"login": "admin", "privilege": "admin"})
        >>> print(token[:20])
        eyJhbGciOiJIUzI1NiIs...
    """
    privilege = privilege or user_data.get('privilege', PRIVILEGE_ENDPOINT)
    
    # Determine expiry based on privilege
    if privilege == PRIVILEGE_ADMIN:
        expiry_hours = JWT_EXPIRY_ADMIN_HOURS
    else:
        expiry_hours = JWT_EXPIRY_ENDPOINT_HOURS
    
    expiry = datetime.utcnow() + timedelta(hours=expiry_hours)
    
    payload = {
        'login': user_data['login'],
        'privilege': privilege,
        'exp': expiry,
        'iat': datetime.utcnow()
    }
    
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    return token


def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token string
    
    Returns:
        Decoded payload dict with 'login' and 'privilege'
    
    Raises:
        HTTPException: If token is invalid or expired (401)
    
    Example:
        >>> payload = verify_token("eyJhbGci...")
        >>> print(payload['login'])
        admin
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}"
        )


def refresh_token(old_token: str) -> str:
    """
    Refresh an existing token (extends expiry).
    
    Args:
        old_token: Current JWT token
    
    Returns:
        New JWT token with extended expiry
    
    Raises:
        HTTPException: If token is invalid (401)
    """
    payload = verify_token(old_token)
    
    # Remove old timestamps
    payload.pop('exp', None)
    payload.pop('iat', None)
    
    # Create new token with same data
    return create_token(payload)


# ==================== Rate Limiting ====================

def check_rate_limit(
    identifier: str,
    operation: str,
    max_requests: Optional[int] = None,
    window_seconds: Optional[int] = None
) -> Tuple[bool, Optional[int]]:
    """
    Check if an operation is within rate limits.
    
    Args:
        identifier: Unique identifier (IP address, username, etc.)
        operation: Operation type (must be in RATE_LIMITS or custom)
        max_requests: Override max requests (uses RATE_LIMITS if None)
        window_seconds: Override time window (uses RATE_LIMITS if None)
    
    Returns:
        Tuple of (allowed: bool, retry_after: Optional[int])
        - allowed: True if within limit, False if exceeded
        - retry_after: Seconds until next allowed request (if exceeded)
    
    Example:
        >>> allowed, retry = check_rate_limit("192.168.1.100", "login")
        >>> if not allowed:
        ...     raise HTTPException(429, f"Rate limit exceeded. Retry after {retry}s")
    """
    # Get rate limit config
    if operation in RATE_LIMITS:
        max_req, window = RATE_LIMITS[operation]
    else:
        max_req = max_requests or 10
        window = window_seconds or 60
    
    now = time()
    key = (identifier, operation)
    
    # Get request history
    history = _rate_limit_storage[key]
    
    # Remove expired entries
    cutoff = now - window
    history[:] = [ts for ts in history if ts > cutoff]
    
    # Check limit
    if len(history) >= max_req:
        # Calculate retry time (when oldest request expires)
        oldest = history[0]
        retry_after = int((oldest + window) - now) + 1
        return False, retry_after
    
    # Add current request
    history.append(now)
    
    return True, None


def record_login_attempt(ip: str, success: bool):
    """
    Record a login attempt (for account lockout protection).
    
    Args:
        ip: Client IP address
        success: Whether login was successful
    
    Notes:
        - Failed attempts are tracked for MAX_LOGIN_ATTEMPTS lockout
        - Successful logins clear the history
    """
    now = time()
    
    if success:
        # Clear failed attempts on success
        _login_attempts[ip] = []
    else:
        # Record failed attempt
        _login_attempts[ip].append((now, False))
        
        # Remove old attempts (older than 5 minutes)
        cutoff = now - 300
        _login_attempts[ip] = [(ts, failed) for ts, failed in _login_attempts[ip] if ts > cutoff]


def is_ip_locked(ip: str) -> Tuple[bool, Optional[int]]:
    """
    Check if an IP is locked out due to failed login attempts.
    
    Args:
        ip: Client IP address
    
    Returns:
        Tuple of (locked: bool, retry_after: Optional[int])
        - locked: True if IP is locked out
        - retry_after: Seconds until unlock (if locked)
    """
    now = time()
    attempts = _login_attempts.get(ip, [])
    
    # Remove old attempts
    cutoff = now - 300  # 5 minutes
    attempts[:] = [(ts, failed) for ts, failed in attempts if ts > cutoff]
    
    # Count failed attempts
    failed_count = sum(1 for _, failed in attempts if failed)
    
    if failed_count >= MAX_LOGIN_ATTEMPTS:
        # Calculate unlock time
        oldest_failed = next((ts for ts, failed in attempts if failed), None)
        if oldest_failed:
            retry_after = int((oldest_failed + 300) - now) + 1
            return True, retry_after
    
    return False, None


# ==================== FastAPI Dependencies ====================

async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """
    FastAPI dependency: Extract and verify JWT token from Authorization header.
    
    Args:
        request: FastAPI request object
        credentials: HTTP Bearer credentials
    
    Returns:
        Decoded token payload (user data)
    
    Raises:
        HTTPException: If token is missing or invalid (401)
    
    Usage:
        @app.get("/protected")
        async def protected_route(user = Depends(get_current_user)):
            return {"message": f"Hello {user['login']}"}
    """
    token = credentials.credentials
    
    # Verify token
    payload = verify_token(token)
    
    # Attach to request state for logging
    request.state.user = payload
    
    return payload


def require_auth(required_privilege: Optional[str] = None):
    """
    FastAPI dependency factory: Require authentication with optional privilege check.
    
    Args:
        required_privilege: Minimum required privilege (PRIVILEGE_ADMIN or PRIVILEGE_ENDPOINT)
                           If None, any authenticated user is allowed
    
    Returns:
        FastAPI dependency function
    
    Raises:
        HTTPException: If unauthorized (401) or insufficient privilege (403)
    
    Usage:
        # Any authenticated user
        @app.get("/users", dependencies=[Depends(require_auth())])
        
        # Admin only
        @app.delete("/users/{user_id}", dependencies=[Depends(require_auth(PRIVILEGE_ADMIN))])
    """
    async def check_auth(user: Dict = Depends(get_current_user)) -> Dict[str, Any]:
        # Check privilege if specified
        if required_privilege:
            if user.get('privilege') != required_privilege:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient privileges. Required: {required_privilege}"
                )
        
        return user
    
    return check_auth


def rate_limit(operation: str):
    """
    FastAPI dependency factory: Enforce rate limiting for an operation.
    
    Args:
        operation: Operation type (must be in RATE_LIMITS or defaults to 10/min)
    
    Returns:
        FastAPI dependency function
    
    Raises:
        HTTPException: If rate limit exceeded (429)
    
    Usage:
        @app.post("/api/verify", dependencies=[Depends(rate_limit("verify"))])
        async def verify_fingerprint(...):
            ...
    """
    async def check_limit(request: Request, user: Dict = Depends(get_current_user)):
        # Use username as identifier (or IP for unauthenticated)
        identifier = user.get('login', request.client.host)
        
        allowed, retry_after = check_rate_limit(identifier, operation)
        
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded for {operation}. Retry after {retry_after} seconds",
                headers={"Retry-After": str(retry_after)}
            )
        
        return user
    
    return check_limit


# ==================== Utility Functions ====================

def get_client_ip(request: Request) -> str:
    """
    Extract client IP address from request (handles proxies).
    
    Args:
        request: FastAPI request object
    
    Returns:
        Client IP address string
    """
    # Check for proxy headers
    forwarded = request.headers.get('X-Forwarded-For')
    if forwarded:
        return forwarded.split(',')[0].strip()
    
    real_ip = request.headers.get('X-Real-IP')
    if real_ip:
        return real_ip
    
    # Fallback to direct connection
    return request.client.host


def cleanup_rate_limit_storage():
    """
    Clean up expired rate limit entries (call periodically).
    
    Removes all entries older than the longest rate limit window.
    """
    now = time()
    max_window = max(window for _, window in RATE_LIMITS.values())
    cutoff = now - max_window
    
    # Clean rate limits
    for key in list(_rate_limit_storage.keys()):
        history = _rate_limit_storage[key]
        history[:] = [ts for ts in history if ts > cutoff]
        
        if not history:
            del _rate_limit_storage[key]
    
    # Clean login attempts
    for ip in list(_login_attempts.keys()):
        attempts = _login_attempts[ip]
        attempts[:] = [(ts, failed) for ts, failed in attempts if ts > cutoff]
        
        if not attempts:
            del _login_attempts[ip]


# ==================== Password Utilities ====================

import bcrypt


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Args:
        password: Plain text password
    
    Returns:
        Bcrypt hash string
    
    Example:
        >>> hashed = hash_password("secret123")
        >>> print(hashed[:7])
        $2b$12$
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """
    Verify a password against a bcrypt hash.
    
    Args:
        password: Plain text password
        hashed: Bcrypt hash string
    
    Returns:
        True if password matches, False otherwise
    
    Example:
        >>> hashed = hash_password("secret123")
        >>> verify_password("secret123", hashed)
        True
        >>> verify_password("wrong", hashed)
        False
    """
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    except Exception as e:
        log_error(e, context="verify_password")
        return False


# ==================== Testing ====================

if __name__ == "__main__":
    print("Testing webserver authentication...")
    
    # Test password hashing
    print("\n1. Password hashing:")
    hashed = hash_password("test123")
    print(f"   Hash: {hashed[:30]}...")
    print(f"   Verify correct: {verify_password('test123', hashed)}")
    print(f"   Verify wrong: {verify_password('wrong', hashed)}")
    
    # Test JWT tokens
    print("\n2. JWT tokens:")
    user_data = {"login": "admin", "privilege": PRIVILEGE_ADMIN}
    token = create_token(user_data)
    print(f"   Token: {token[:50]}...")
    
    payload = verify_token(token)
    print(f"   Decoded: login={payload['login']}, privilege={payload['privilege']}")
    
    # Test rate limiting
    print("\n3. Rate limiting:")
    for i in range(12):
        allowed, retry = check_rate_limit("test_user", "verify", max_requests=10, window_seconds=60)
        if allowed:
            print(f"   Request {i+1}: ✓ Allowed")
        else:
            print(f"   Request {i+1}: ✗ Rate limit exceeded (retry in {retry}s)")
    
    # Test login lockout
    print("\n4. Login lockout:")
    test_ip = "192.168.1.100"
    for i in range(7):
        locked, retry = is_ip_locked(test_ip)
        if locked:
            print(f"   Attempt {i+1}: ✗ IP locked (unlock in {retry}s)")
        else:
            print(f"   Attempt {i+1}: ✓ Not locked")
            record_login_attempt(test_ip, success=False)
    
    print("\n✓ All tests completed")
