"""
Authentication Routes
Login, token refresh, logout endpoints.
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel

from ..database import BiometricDatabase
from ..auth import create_token, get_client_ip, is_ip_locked, record_login_attempt, get_current_user
from ..logger import log_auth
from ..config import PRIVILEGE_ADMIN, PRIVILEGE_ENDPOINT


router = APIRouter(tags=["Authentication"])


class LoginRequest(BaseModel):
    """Login request model."""
    login: str
    password: str


class LoginResponse(BaseModel):
    """Login response model."""
    token: str
    login: str
    privilege: str
    expires_in: int


@router.post("/login", response_model=LoginResponse)
async def login(request: Request, req: LoginRequest):
    """
    Authenticate user and return JWT token.
    
    Request:
        - login: Username
        - password: Password
    
    Response:
        - token: JWT token
        - login: Username
        - privilege: User privilege (admin/endpoint)
        - expires_in: Token expiry in seconds
    """
    # DEBUG: Log received data
    print(f"DEBUG LOGIN: Received login='{req.login}', password={'*' * len(req.password)}")
    
    ip = get_client_ip(request)
    
    # Check IP lockout
    locked, retry = is_ip_locked(ip)
    if locked:
        log_auth("LOGIN", req.login, ip, success=False, details=f"IP locked (retry in {retry}s)")
        raise HTTPException(
            status_code=429,
            detail=f"Too many failed login attempts. Try again in {retry} seconds",
            headers={"Retry-After": str(retry)}
        )
    
    # Authenticate
    db = BiometricDatabase()
    user_data = db.authenticate(req.login, req.password)
    
    if not user_data:
        # Record failed attempt
        record_login_attempt(ip, success=False)
        log_auth("LOGIN", req.login, ip, success=False, details="Invalid credentials")
        
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )
    
    # Success - clear failed attempts
    record_login_attempt(ip, success=True)
    
    # Create token
    token = create_token(user_data)
    
    log_auth("LOGIN", req.login, ip, success=True)
    
    # Get expiry from config
    from ..config import JWT_EXPIRY_ADMIN_HOURS, JWT_EXPIRY_ENDPOINT_HOURS
    
    if user_data['privilege'] == PRIVILEGE_ADMIN:
        expires_in = JWT_EXPIRY_ADMIN_HOURS * 3600
    else:
        expires_in = JWT_EXPIRY_ENDPOINT_HOURS * 3600
    
    return LoginResponse(
        token=token,
        login=user_data['login'],
        privilege=user_data['privilege'],
        expires_in=expires_in
    )


@router.post("/refresh")
async def refresh_token(request: Request, user = Depends(get_current_user)):
    """
    Refresh JWT token (extend expiry).
    
    Requires valid JWT token in Authorization header.
    
    Response:
        - token: New JWT token with extended expiry
    """
    from ..auth import refresh_token as refresh_jwt
    
    # Get old token
    auth_header = request.headers.get("Authorization", "")
    old_token = auth_header.replace("Bearer ", "")
    
    # Create new token
    new_token = refresh_jwt(old_token)
    
    ip = get_client_ip(request)
    log_auth("TOKEN_REFRESH", user['login'], ip, success=True)
    
    return {"token": new_token}


@router.post("/logout")
async def logout(request: Request, user = Depends(get_current_user)):
    """
    Logout user (for logging purposes - JWT is stateless).
    
    Note: Since JWT is stateless, this just logs the event.
    Client should discard the token.
    """
    ip = get_client_ip(request)
    log_auth("LOGOUT", user['login'], ip, success=True)
    
    return {"message": "Logged out successfully"}
