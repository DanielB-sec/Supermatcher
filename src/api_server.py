"""FastAPI Server for Fingerprint Identification System.

RESTful API with:
- TLS/HTTPS encryption
- HTTP Basic Authentication
- Role-based access control (admin/endpoint)
- Fingerprint identification and verification
- User enrollment and management
- SQLCipher encrypted database
- Rotating file logs

Endpoints:
- POST /api/v1/identify - Identify fingerprint (1:N)
- POST /api/v1/verify - Verify fingerprint (1:1)
- POST /api/v1/enroll - Enroll new user (admin only)
- POST /api/v1/enroll/{user_id}/add-samples - Add samples to existing user (admin only)
- DELETE /api/v1/users/{user_id} - Delete user (admin only)
- GET /api/v1/users - List all users (admin only)
- GET /docs - Interactive API documentation (Swagger UI)
"""

import argparse
import getpass
import time
from pathlib import Path
from typing import List, Optional
import sys
import subprocess

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2

# Import our modules
from db_manager import DatabaseManager
from auth_manager import AuthManager, parse_basic_auth
from logging_config import (
    setup_logging, log_identify, log_verify, log_enroll, log_delete,
    log_auth_failure, log_permission_denied, log_quality_rejected,
    log_server_start, log_server_shutdown, log_database_init
)

# Import fingerprint matcher from v0.7
# Note: Assuming supermatcher_v0_7.py exists (renamed from supermatcher_v0.7.py)
try:
    import supermatcher_v1_0 as matcher
    from supermatcher_v1_0 import (
        FingerprintMatcher, FingerprintTemplate, CancelableHasher,
        FingerprintPipeline, process_single_template
    )
    print("✓ Fingerprint matcher imported successfully")
except ImportError as e:
    print(f"⚠️  WARNING: Failed to import supermatcher_v0_7: {e}")
    print("   Make sure supermatcher_v0_7.py exists in src/ folder")
    print(f"   Current working directory: {Path.cwd()}")
    print(f"   sys.path: {sys.path[:3]}")
    sys.exit(1)  # Cannot run without matcher

# Constants
QUALITY_THRESHOLD = 0.30
MIN_SAMPLES_ENROLL = 5
API_VERSION = "1.0.0"

# Global instances (initialized in main)
app = FastAPI(
    title="Fingerprint Identification API",
    description="Secure biometric identification system with TLS and role-based access",
    version=API_VERSION
)

# Security
security = HTTPBasic()

# Database and auth managers
db_manager: Optional[DatabaseManager] = None
auth_manager: Optional[AuthManager] = None
logger = None
fingerprint_matcher: Optional[FingerprintMatcher] = None
hasher: Optional[CancelableHasher] = None


# ============================================================================
# AUTHENTICATION DEPENDENCY
# ============================================================================

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)) -> tuple:
    """Authenticate user from HTTP Basic Auth.
    
    Returns:
        Tuple of (username, role)
        
    Raises:
        HTTPException 401 if authentication fails
    """
    authenticated, role = auth_manager.authenticate(
        credentials.username, 
        credentials.password
    )
    
    if not authenticated:
        log_auth_failure(logger, credentials.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    return credentials.username, role


def require_role(required_role: str):
    """Dependency to check user role.
    
    Args:
        required_role: 'admin' or 'endpoint'
        
    Returns:
        Function that checks role
    """
    def check_role(user_info: tuple = Depends(get_current_user)) -> tuple:
        username, role = user_info
        
        if not auth_manager.has_permission(role, required_role):
            log_permission_denied(logger, username, "OPERATION", required_role, role)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires {required_role} role"
            )
        
        return username, role
    
    return check_role


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_image_from_upload(file: UploadFile) -> Optional[np.ndarray]:
    """Load image from uploaded file.
    
    Args:
        file: UploadFile from FastAPI
        
    Returns:
        Numpy array (grayscale image) or None if invalid
    """
    try:
        # Read file content
        contents = file.file.read()
        
        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return None
        
        return img
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        return None
    finally:
        file.file.close()


def process_and_validate_image(img: np.ndarray, min_quality: float = QUALITY_THRESHOLD, identifier: str = "probe") -> tuple:
    """Process image and validate quality.
    
    Args:
        img: Input grayscale image
        min_quality: Minimum quality threshold
        identifier: Template identifier
        
    Returns:
        Tuple of (template: FingerprintTemplate, quality: float) or (None, 0.0) if invalid
    """
    import tempfile
    try:
        # Save image to temporary file (FingerprintPipeline.process needs a Path)
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            cv2.imwrite(str(tmp_path), img)
        
        try:
            # Create pipeline instance and process
            pipeline = FingerprintPipeline(include_level3=False, hasher=hasher)
            template = pipeline.process(tmp_path, identifier=identifier)
            
            if template is None:
                return None, 0.0
            
            quality = template.quality
            
            # Validate quality
            if quality < min_quality:
                return None, quality
            
            return template, quality
        finally:
            # Clean up temporary file
            if tmp_path.exists():
                tmp_path.unlink()
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None, 0.0


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - redirect to docs or client."""
    return """
    <html>
        <head><title>Fingerprint API</title></head>
        <body style="font-family: Arial; background: #1a1a1a; color: #fff; padding: 50px;">
            <h1>🔐 Fingerprint Identification API</h1>
            <p>Version: {}</p>
            <ul>
                <li><a href="/docs" style="color: #4CAF50;">📚 API Documentation (Swagger)</a></li>
                <li><a href="/client.html" style="color: #4CAF50;">🌐 Web Client</a></li>
            </ul>
        </body>
    </html>
    """.format(API_VERSION)


@app.post("/api/v1/identify")
async def identify(
    image: UploadFile = File(...),
    user_info: tuple = Depends(require_role('endpoint'))
):
    """Identify fingerprint (1:N matching).
    
    Args:
        image: Fingerprint image file
        user_info: Authenticated user (from dependency)
        
    Returns:
        JSON with match result
    """
    username, role = user_info
    start_time = time.time()
    
    # Load and validate image
    img = load_image_from_upload(image)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    # Process image
    probe_template, quality = process_and_validate_image(img)
    
    if probe_template is None:
        if quality > 0:
            log_quality_rejected(logger, username, quality, QUALITY_THRESHOLD)
            raise HTTPException(
                status_code=422, 
                detail=f"Quality too low: {quality:.3f} < {QUALITY_THRESHOLD}"
            )
        raise HTTPException(status_code=422, detail="Failed to extract features")
    
    # Load all templates from database
    templates = db_manager.get_all_fingerprints()
    
    if not templates:
        raise HTTPException(status_code=404, detail="No enrolled users in database")
    
    # Create matcher and identify
    matcher_instance = FingerprintMatcher(templates, hasher)
    
    try:
        results = matcher_instance.identify(probe_template, top_k=3)
        
        if results and len(results) > 0:
            top_match = results[0]
            user_id = int(top_match[0])  # (identifier, score)
            confidence = float(top_match[1])
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            log_identify(logger, username, user_id, confidence, processing_time_ms, success=True)
            
            return {
                "status": "success",
                "user_id": user_id,
                "confidence": confidence,
                "processing_time_ms": processing_time_ms,
                "quality": quality
            }
        else:
            processing_time_ms = int((time.time() - start_time) * 1000)
            log_identify(logger, username, None, 0.0, processing_time_ms, success=False)
            
            return {
                "status": "no_match",
                "user_id": None,
                "confidence": 0.0,
                "processing_time_ms": processing_time_ms
            }
    
    except Exception as e:
        logger.error(f"Error during identification: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/verify")
async def verify(
    image: UploadFile = File(...),
    claimed_id: int = Form(...),
    user_info: tuple = Depends(require_role('endpoint'))
):
    """Verify fingerprint against claimed identity (1:1 matching).
    
    Args:
        image: Fingerprint image file
        claimed_id: User ID to verify against
        user_info: Authenticated user
        
    Returns:
        JSON with verification result
    """
    username, role = user_info
    start_time = time.time()
    
    # Load and validate image
    img = load_image_from_upload(image)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    # Process image
    probe_template, quality = process_and_validate_image(img)
    
    if probe_template is None:
        if quality > 0:
            log_quality_rejected(logger, username, quality, QUALITY_THRESHOLD)
            raise HTTPException(
                status_code=422, 
                detail=f"Quality too low: {quality:.3f} < {QUALITY_THRESHOLD}"
            )
        raise HTTPException(status_code=422, detail="Failed to extract features")
    
    # Get claimed user's template
    claimed_template = db_manager.get_fingerprint(claimed_id)
    
    if claimed_template is None:
        raise HTTPException(status_code=404, detail=f"User {claimed_id} not found")
    
    # Verify
    try:
        matcher_instance = FingerprintMatcher([claimed_template], hasher)
        is_match, confidence = matcher_instance.verify(probe_template, str(claimed_id))
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        log_verify(logger, username, claimed_id, is_match, confidence, processing_time_ms)
        
        return {
            "status": "match" if is_match else "no_match",
            "claimed_id": claimed_id,
            "is_match": is_match,
            "confidence": confidence,
            "processing_time_ms": processing_time_ms,
            "quality": quality
        }
    
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/enroll")
async def enroll(
    user_id: int = Form(...),
    name: str = Form(""),
    images: List[UploadFile] = File(...),
    user_info: tuple = Depends(require_role('admin'))
):
    """Enroll new user with fingerprint samples.
    
    Args:
        user_id: Unique user identifier
        name: Optional user name
        images: List of fingerprint image files (minimum 5)
        user_info: Authenticated admin user
        
    Returns:
        JSON with enrollment result
    """
    username, role = user_info
    
    # Check if user already exists
    if db_manager.user_exists(user_id):
        raise HTTPException(status_code=409, detail=f"User {user_id} already exists")
    
    # Validate number of samples
    if len(images) < MIN_SAMPLES_ENROLL:
        raise HTTPException(
            status_code=400,
            detail=f"Minimum {MIN_SAMPLES_ENROLL} images required, got {len(images)}"
        )
    
    # Process all images
    templates_list = []
    qualities = []
    
    for idx, image_file in enumerate(images):
        img = load_image_from_upload(image_file)
        
        if img is None:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image format: {image_file.filename}"
            )
        
        template, quality = process_and_validate_image(img)
        
        if template is None:
            if quality > 0:
                log_quality_rejected(logger, username, quality, QUALITY_THRESHOLD)
                raise HTTPException(
                    status_code=422,
                    detail=f"Image {idx+1} quality too low: {quality:.3f} < {QUALITY_THRESHOLD}"
                )
            raise HTTPException(
                status_code=422,
                detail=f"Failed to extract features from image {idx+1}"
            )
        
        templates_list.append(template)
        qualities.append(quality)
    
    # Merge templates into single user template
    # (This uses the build_template logic from supermatcher_v0.7)
    # For now, we'll use the first template and update identifier
    merged_template = templates_list[0]
    merged_template.identifier = str(user_id)
    
    # TODO: Implement proper template merging if needed
    # For v0.7, multiple samples are typically processed into one template already
    
    avg_quality = np.mean(qualities)
    
    # Save to database
    success = db_manager.add_fingerprint(
        user_id=user_id,
        template=merged_template,
        name=name,
        num_samples=len(images),
        avg_quality=avg_quality
    )
    
    if not success:
        log_enroll(logger, username, user_id, len(images), avg_quality, success=False)
        raise HTTPException(status_code=500, detail="Failed to save template to database")
    
    log_enroll(logger, username, user_id, len(images), avg_quality, success=True)
    
    return {
        "status": "success",
        "user_id": user_id,
        "name": name,
        "samples_processed": len(images),
        "avg_quality": round(avg_quality, 3),
        "individual_qualities": [round(q, 3) for q in qualities]
    }


@app.post("/api/v1/enroll/{user_id}/add-samples")
async def add_samples(
    user_id: int,
    images: List[UploadFile] = File(...),
    mode: str = Form("merge"),  # "merge" or "rebuild"
    user_info: tuple = Depends(require_role('admin'))
):
    """Add more samples to existing user template.
    
    Args:
        user_id: Existing user identifier
        images: Additional fingerprint images
        mode: "merge" (default) or "rebuild"
        user_info: Authenticated admin user
        
    Returns:
        JSON with update result
    """
    username, role = user_info
    
    # Check if user exists
    existing_template = db_manager.get_fingerprint(user_id)
    
    if existing_template is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    
    # Process new images
    new_templates = []
    qualities = []
    
    for idx, image_file in enumerate(images):
        img = load_image_from_upload(image_file)
        
        if img is None:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image format: {image_file.filename}"
            )
        
        template, quality = process_and_validate_image(img)
        
        if template is None:
            if quality > 0:
                raise HTTPException(
                    status_code=422,
                    detail=f"Image {idx+1} quality too low: {quality:.3f}"
                )
            raise HTTPException(status_code=422, detail=f"Failed to process image {idx+1}")
        
        new_templates.append(template)
        qualities.append(quality)
    
    # Merge or rebuild
    if mode == "merge":
        # Simple merge: combine features (placeholder - actual implementation depends on template structure)
        merged_template = existing_template
        # TODO: Implement actual merging logic based on FingerprintTemplate structure
    else:
        # Rebuild: use new templates only
        merged_template = new_templates[0]
    
    merged_template.identifier = str(user_id)
    
    # Get existing metadata
    users = db_manager.list_users()
    existing_user = next((u for u in users if u['id'] == user_id), None)
    old_num_samples = existing_user['num_samples'] if existing_user else 0
    
    new_num_samples = old_num_samples + len(images) if mode == "merge" else len(images)
    avg_quality = np.mean(qualities)
    
    # Update database
    success = db_manager.update_fingerprint(
        user_id=user_id,
        template=merged_template,
        num_samples=new_num_samples,
        avg_quality=avg_quality
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update template")
    
    logger.info(
        f"Added {len(images)} samples to user {user_id} (mode={mode})",
        extra={'user': username, 'action': 'ADD_SAMPLES'}
    )
    
    return {
        "status": "success",
        "user_id": user_id,
        "mode": mode,
        "new_samples": len(images),
        "total_samples": new_num_samples,
        "avg_quality": round(avg_quality, 3)
    }


@app.delete("/api/v1/users/{user_id}")
async def delete_user(
    user_id: int,
    user_info: tuple = Depends(require_role('admin'))
):
    """Delete user from database.
    
    Args:
        user_id: User identifier to delete
        user_info: Authenticated admin user
        
    Returns:
        JSON with deletion result
    """
    username, role = user_info
    
    success = db_manager.delete_fingerprint(user_id)
    
    if not success:
        log_delete(logger, username, user_id, success=False)
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    
    log_delete(logger, username, user_id, success=True)
    
    return {
        "status": "success",
        "user_id": user_id,
        "message": "User deleted successfully"
    }


@app.get("/api/v1/users")
async def list_users(
    user_info: tuple = Depends(require_role('admin'))
):
    """List all enrolled users.
    
    Args:
        user_info: Authenticated admin user
        
    Returns:
        JSON with list of users
    """
    username, role = user_info
    
    users = db_manager.list_users()
    
    logger.info(
        f"Listed {len(users)} users",
        extra={'user': username, 'action': 'LIST_USERS'}
    )
    
    return {
        "status": "success",
        "users": users,
        "total": len(users)
    }


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint (no auth required)."""
    return {
        "status": "healthy",
        "version": API_VERSION,
        "database": "connected" if db_manager and db_manager.conn else "disconnected"
    }


# ============================================================================
# STARTUP / SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize components on server startup."""
    global db_manager, auth_manager, logger, hasher
    
    logger.info("Starting fingerprint API server...")
    
    # Connect to database
    db_manager.connect()
    
    # Create hasher with default parameters
    # feature_dim=736, projection_dim=512 are the defaults from supermatcher_v0_7.py
    hasher = CancelableHasher(
        feature_dim=736,  # FEATURE_VECTOR_DIM
        projection_dim=512,  # DEFAULT_PROJECTION_DIM
        key="TB_SB_API_DEFAULT_KEY_2025",  # Default key (should be changed per user in production)
        hash_count=1
    )
    
    logger.info("Server startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown."""
    log_server_shutdown(logger)
    
    if db_manager:
        db_manager.close()


# ============================================================================
# INITIALIZATION AND CLI
# ============================================================================

def generate_self_signed_cert(cert_dir: Path):
    """Generate self-signed TLS certificate.
    
    Args:
        cert_dir: Directory to save certificate files
    """
    cert_dir.mkdir(parents=True, exist_ok=True)
    
    cert_file = cert_dir / "server.crt"
    key_file = cert_dir / "server.key"
    
    if cert_file.exists() and key_file.exists():
        print(f"✓ Certificates already exist in {cert_dir}")
        return
    
    print("Generating self-signed TLS certificate...")
    
    try:
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:4096", "-nodes",
            "-keyout", str(key_file),
            "-out", str(cert_file),
            "-days", "365",
            "-subj", "/CN=localhost/O=TB_SB/C=PT"
        ], check=True, capture_output=True)
        
        print(f"✓ Certificate generated: {cert_file}")
        print(f"✓ Private key generated: {key_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating certificate: {e}")
        print("   Ensure OpenSSL is installed and in PATH")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ OpenSSL not found in PATH")
        print("   Install OpenSSL or generate certificates manually")
        sys.exit(1)


def initialize_database(db_path: str):
    """Initialize database with tables and default admin user.
    
    Args:
        db_path: Path to database file
    """
    global db_manager, auth_manager, logger
    
    print("\n" + "="*60)
    print("INITIALIZING FINGERPRINT API DATABASE")
    print("="*60 + "\n")
    
    # Setup logging first
    logger = setup_logging()
    
    # Create database manager
    db_manager = DatabaseManager(db_path)
    db_manager.connect()
    
    # Create tables
    print("Creating database tables...")
    db_manager.initialize_database()
    
    # Create auth manager
    auth_manager = AuthManager(db_manager)
    
    # Create default admin user
    print("\nCreating default admin user...")
    print("Username: admin")
    
    while True:
        password = getpass.getpass("Enter admin password: ")
        password_confirm = getpass.getpass("Confirm password: ")
        
        if password == password_confirm:
            if len(password) < 6:
                print("❌ Password too short (minimum 6 characters)")
                continue
            break
        else:
            print("❌ Passwords don't match, try again")
    
    success = auth_manager.create_user("admin", password, "admin")
    
    if success:
        print("✓ Admin user created successfully")
    else:
        print("⚠️  Admin user already exists")
    
    # Generate TLS certificates
    cert_dir = Path("certs")
    generate_self_signed_cert(cert_dir)
    
    log_database_init(logger)
    
    db_manager.close()
    
    print("\n" + "="*60)
    print("INITIALIZATION COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("  1. Start the server: python src/api_server.py")
    print("  2. Open web client: https://localhost:8443/client.html")
    print("  3. Or use API docs: https://localhost:8443/docs")
    print("\n")


def main():
    """Main entry point with CLI argument parsing."""
    global db_manager, auth_manager, logger
    
    parser = argparse.ArgumentParser(description="Fingerprint API Server")
    
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize database and create admin user"
    )
    
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind server (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8443,
        help="Port to bind server (default: 8443)"
    )
    
    parser.add_argument(
        "--db-path",
        default="data/fingerprints.db",
        help="Path to SQLCipher database (default: data/fingerprints.db)"
    )
    
    parser.add_argument(
        "--cert",
        default="certs/server.crt",
        help="Path to TLS certificate (default: certs/server.crt)"
    )
    
    parser.add_argument(
        "--key",
        default="certs/server.key",
        help="Path to TLS private key (default: certs/server.key)"
    )
    
    parser.add_argument(
        "--no-ssl",
        action="store_true",
        help="Run without SSL/TLS (NOT RECOMMENDED for production)"
    )
    
    args = parser.parse_args()
    
    # Initialize database if requested
    if args.init_db:
        initialize_database(args.db_path)
        return
    
    # Setup logging
    logger = setup_logging()
    
    # Initialize database manager
    db_manager = DatabaseManager(args.db_path)
    auth_manager = AuthManager(db_manager)
    
    # Check if certificates exist
    cert_path = Path(args.cert)
    key_path = Path(args.key)
    
    if not args.no_ssl:
        if not cert_path.exists() or not key_path.exists():
            print("❌ TLS certificates not found!")
            print(f"   Certificate: {cert_path}")
            print(f"   Private key: {key_path}")
            print("\nRun: python src/api_server.py --init-db")
            print("Or use --no-ssl flag (NOT RECOMMENDED)")
            sys.exit(1)
    
    # Configure server
    ssl_config = None
    protocol = "HTTP"
    
    if not args.no_ssl:
        ssl_config = {
            "ssl_keyfile": str(key_path),
            "ssl_certfile": str(cert_path)
        }
        protocol = "HTTPS"
    
    log_server_start(logger, args.host, args.port, ssl=not args.no_ssl)
    
    print("\n" + "="*60)
    print(f"🔐 FINGERPRINT API SERVER v{API_VERSION}")
    print("="*60)
    print(f"\n  URL: {protocol.lower()}://{args.host}:{args.port}")
    print(f"  Docs: {protocol.lower()}://{args.host}:{args.port}/docs")
    print(f"  Client: {protocol.lower()}://{args.host}:{args.port}/client.html")
    print(f"\n  Database: {args.db_path}")
    print(f"  Logs: data/logs/server.log")
    
    if args.no_ssl:
        print("\n  ⚠️  WARNING: Running without TLS encryption!")
    
    print("\n  Press CTRL+C to stop\n")
    print("="*60 + "\n")
    
    # Start server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        **ssl_config if ssl_config else {}
    )


if __name__ == "__main__":
    main()
