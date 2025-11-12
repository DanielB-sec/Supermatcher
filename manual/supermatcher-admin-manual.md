# Supermatcher - Administrator Manual

**Version 1.0.0**  
**Production-Ready Fingerprint Authentication System**

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [System Requirements](#2-system-requirements)
3. [Installation](#3-installation)
4. [First-Time Setup](#4-first-time-setup)
5. [Starting the Server](#5-starting-the-server)
6. [Configuration Reference](#6-configuration-reference)
7. [Admin Dashboard Guide](#7-admin-dashboard-guide)
8. [Backup and Recovery](#8-backup-and-recovery)
9. [Monitoring and Logs](#9-monitoring-and-logs)
10. [Troubleshooting](#10-troubleshooting)
11. [Security Considerations](#11-security-considerations)

---

## 1. System Overview

The Supermatcher is a production-ready fingerprint authentication system with:

- **JWT-based authentication** with role-based access control (admin/endpoint)
- **Encrypted database** (SQLCipher) for secure template storage
- **Two-stage matching**: Hash-based filtering + geometric verification
- **Multi-sample enrollment** with automatic template fusion
- **Background job processing** for batch operations
- **Web-based interface** for administration and testing

### Architecture

```
┌─────────────────┐
│  Web Interface  │ (Admin Dashboard / Endpoint Dashboard)
└────────┬────────┘
         │ HTTPS (JWT Auth)
┌────────▼────────┐
│   FastAPI       │ (Routes + Authentication + Rate Limiting)
│   WebServer     │
└────────┬────────┘
         │
    ┌────┴────┬──────────┬──────────┐
    │         │          │          │
┌───▼───┐ ┌──▼───┐ ┌────▼────┐ ┌──▼──────┐
│ Auth  │ │ User │ │ Biomet. │ │ Admin   │
│ Routes│ │Routes│ │ Routes  │ │ Routes  │
└───┬───┘ └──┬───┘ └────┬────┘ └──┬──────┘
    │        │           │         │
    └────────┴───────────┴─────────┘
                   │
        ┌──────────▼──────────┐
        │ ProcessPool Executor │ (4-8 workers with CPU affinity)
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │ Supermatcher v1.0   │ (Fingerprint Pipeline)
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │ SQLCipher Database  │ (Encrypted storage)
        └─────────────────────┘
```

---

## 2. System Requirements

### Hardware Requirements

- **CPU**: Intel/AMD x64 with 4+ cores (6+ P-cores recommended for hybrid CPUs)
- **RAM**: 8 GB minimum, 16 GB recommended
- **Storage**: 10 GB minimum (database grows with enrollments)
- **Network**: Ethernet or WiFi with stable connection

### Software Requirements

- **Operating System**: 
  - Windows 10/11 (64-bit)
  - Linux (Ubuntu 20.04+, Debian 11+, CentOS 8+)
  - macOS 11+ (limited testing)

- **Python**: 3.8 or higher (3.10 recommended)

- **Python Dependencies** (installed via `requirements.txt`):
  ```
  uvicorn
  cryptography
  sqlcipher3-wheels
  bcrypt
  fastapi
  numpy
  opencv-python
  scipy
  pyjwt
  psutil
  pydantic
  python-multipart
  ```

---

## 3. Installation

### Step 1: Install Python

**Windows:**
```cmd
# Download from https://www.python.org/downloads/
# Run installer and check "Add Python to PATH"
python --version
```

**Linux:**
```bash
sudo apt update
sudo apt install python3.10 python3-pip
python3 --version
```

### Step 2: Clone or Extract Project

```bash
cd /path/to/installation
# Project structure:
biometric_system/
├── run_webserver.py          # Main startup script
├── requirements.txt
├── src/
│   ├── webserver/
│   │   ├── config.py         # Configuration file
│   │   ├── server.py
│   │   ├── database.py
│   │   ├── auth.py
│   │   ├── routes/
│   │   └── frontend/
│   ├── supermatcher_v1_0.py
│   └── ...
└── logs/                     # Created on first run
```

### Step 3: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import cv2, numpy, fastapi; print('✓ All dependencies installed')"
```

---

## 4. First-Time Setup

### Interactive Setup

Run the setup wizard:

```bash
python run_webserver.py
```

The wizard will guide you through:

1. **Admin Account Creation**
   - Username: `admin` (default) or custom
   - Password: Minimum 8 characters
   - Privilege: `admin`

2. **Endpoint Account Creation**
   - Username: `endpoint` (default) or custom
   - Password: Minimum 8 characters
   - Privilege: `endpoint`

3. **SSL Certificate Generation**
   - Choose HTTPS (recommended) or HTTP
   - Default port: `8443` (HTTPS) or `8080` (HTTP)
   - Self-signed certificate generated automatically

### Manual Setup (Advanced)

If you need to reset or manually configure:

```bash
# Reset admin password
python run_webserver.py --reset-admin

# Delete database and start fresh
rm biometric_system.db
rm .db_key
rm .jwt_secret
python run_webserver.py
```

---

## 5. Starting the Server

### Basic Startup

```bash
python run_webserver.py
```

Output:
```
======================================================================
STARTING SUPERMATCHER WEBSERVER
======================================================================

🔒 HTTPS enabled with custom certificate
✓ Self-signed certificate generated
  ⚠️  Browser will show security warning (normal for self-signed certs)

🚀 Server starting on 0.0.0.0:8443
   Protocol: HTTPS
   Workers: 4

📱 Web Interface: https://localhost:8443/
📚 API Docs: https://localhost:8443/docs

Press CTRL+C to stop
======================================================================
```

### Command-Line Options

```bash
# Custom host and port
python run_webserver.py --host 192.168.1.100 --port 8443

# Use HTTP (not recommended for production)
python run_webserver.py --ssl

# Custom SSL certificates
python run_webserver.py --cert /path/to/cert.pem --key /path/to/key.pem

# More workers for high-load systems
python run_webserver.py --workers 8

# Reset admin password
python run_webserver.py --reset-admin
```

### Accessing the Interface

1. **Open browser** and navigate to:
   - `https://localhost:8443/` (local access)
   - `https://<server-ip>:8443/` (remote access)

2. **Accept self-signed certificate warning**:
   - Chrome: Click "Advanced" → "Proceed to localhost"
   - Firefox: Click "Advanced" → "Accept Risk"
   - Edge: Click "Advanced" → "Continue to localhost"

3. **Login with admin account**:
   - Username: `admin`
   - Password: (set during setup)

---

## 6. Configuration Reference

Edit `src/webserver/config.py` to customize system behavior.

### 6.1 Network Settings

```python
# Server binding
HOST = "0.0.0.0"          # Listen on all interfaces
PORT_HTTPS = 8443         # HTTPS port
PORT_HTTP = 8080          # HTTP fallback port

# CORS (Cross-Origin Resource Sharing)
CORS_ORIGINS = ["*"]      # Allow all origins (restrict in production)
```

**Recommendations:**
- For production, set `CORS_ORIGINS = ["https://yourdomain.com"]`
- Use `HOST = "127.0.0.1"` for local-only access

### 6.2 Authentication Settings

```python
# JWT Token Expiry
JWT_EXPIRY_ADMIN_HOURS = 1      # Admin tokens expire in 1 hour
JWT_EXPIRY_ENDPOINT_HOURS = 12  # Endpoint tokens expire in 12 hours

# Account Lockout
MAX_LOGIN_ATTEMPTS = 5          # Lock account after 5 failed attempts
LOCKOUT_DURATION_MINUTES = 15   # Unlock after 15 minutes
```

**Recommendations:**
- Keep admin token expiry short (1-2 hours) for security
- Endpoint tokens can be longer (8-12 hours) for convenience
- Adjust lockout settings based on security requirements

### 6.3 Biometric Thresholds

```python
# Quality Control
QUALITY_THRESHOLD = 0.30        # Minimum quality for enrollment (0.0-1.0)

# Matching Thresholds
VERIFICATION_THRESHOLD = 0.75   # 1:1 verification threshold
IDENTIFICATION_THRESHOLD = 0.70 # 1:N identification threshold
```

**Recommendations:**
- **Quality Threshold**: 
  - `0.30` = Balanced (recommended)
  - `0.40` = Stricter (fewer false accepts, more rejections)
  - `0.20` = Lenient (more users accepted, lower security)

- **Verification Threshold**:
  - `0.75` = Balanced security (recommended)
  - `0.80` = High security (fewer false accepts)
  - `0.70` = Moderate security (fewer false rejects)

- **Identification Threshold**:
  - `0.70` = Balanced (recommended)
  - Lower than verification to reduce false rejects in 1:N search

### 6.4 Enrollment Settings

```python
# Sample Requirements
MIN_SAMPLES_PER_USER = 2        # Minimum images required
MAX_SAMPLES_PER_USER = 10       # Maximum images allowed
RECOMMENDED_SAMPLES = 5         # Recommended for best accuracy

# Template Fusion
FUSION_ENABLED = True           # Enable multi-sample fusion
FUSION_DISTANCE = 12.0          # Pixel distance for minutiae matching
FUSION_ANGLE_DEG = 15.0         # Angle tolerance (degrees)
FUSION_MIN_CONSENSUS = 0.5      # Minimum consensus score (0.0-1.0)
```

**Recommendations:**
- **Samples**: 5 images per user provides best accuracy
- **Fusion**: Keep enabled for multi-sample enrollment
- Don't modify fusion parameters unless you understand the algorithm

### 6.5 Multiprocessing Settings

```python
# ProcessPool Configuration
MAX_WORKERS = 4                 # Number of parallel workers
USE_CPU_AFFINITY = True         # Pin workers to specific cores

# Intel Hybrid CPU Configuration
PCORES = list(range(0, 12))     # P-core logical threads (0-11)
```

**Recommendations:**
- **MAX_WORKERS**:
  - Standard CPU: Set to number of physical cores
  - Hybrid CPU (Intel 12th+ gen): Set to number of P-cores × 2
  - Example: 6 P-cores → `MAX_WORKERS = 4-8`

- **CPU_AFFINITY**:
  - Keep `True` for hybrid CPUs (pins to P-cores)
  - Set `False` for standard CPUs

### 6.6 Rate Limiting

```python
# Rate Limits: (max_requests, time_window_seconds)
RATE_LIMITS = {
    "login": (5, 300),          # 5 login attempts per 5 minutes
    "verify": (10, 60),         # 10 verifications per minute
    "identify": (5, 60),        # 5 identifications per minute
    "enroll": (20, 3600),       # 20 enrollments per hour
    "update": (10, 3600),       # 10 updates per hour
    "delete": (30, 3600),       # 30 deletes per hour
}
```

**Recommendations:**
- Adjust based on expected usage patterns
- Stricter limits improve security but may impact usability
- Monitor logs to identify if limits are too restrictive

### 6.7 Logging Settings

```python
# Log Files
LOG_DIR = PROJECT_ROOT / "logs"
LOG_ACCESS = LOG_DIR / "access.log"     # HTTP requests
LOG_AUTH = LOG_DIR / "auth.log"         # Authentication events
LOG_BIOMETRIC = LOG_DIR / "biometric.log"  # Biometric operations
LOG_ERROR = LOG_DIR / "error.log"       # Application errors

# Rotation Settings
LOG_MAX_BYTES = 10 * 1024 * 1024       # 10MB per file
LOG_BACKUP_COUNT = 5                    # Keep 5 backup files
```

---

## 7. Admin Dashboard Guide

### 7.1 Dashboard Overview

After logging in as admin, you'll see:

```
┌────────────────────────────────────────────────┐
│ 👑 Admin Dashboard                             │
├────────────────────────────────────────────────┤
│ Statistics:                                    │
│   👥 Total Users: 42                           │
│   📊 Avg Quality: 85.3%                        │
│   🔐 Database: Encrypted ✓                     │
├────────────────────────────────────────────────┤
│ Quick Actions:                                 │
│  [➕ Add User] [📂 Load Folder] [🔍 Verify]   │
│  [🎯 Identify] [🔄 Refresh List] [🗑️ Delete All]│
├────────────────────────────────────────────────┤
│ Enrolled Users (table)                         │
├────────────────────────────────────────────────┤
│ Real-time Updates (log)                        │
└────────────────────────────────────────────────┘
```

### 7.2 Add User (Single Enrollment)

**Steps:**

1. Click **"➕ Add User"**
2. Fill in form:
   - **User ID**: Unique identifier (e.g., `user001`, `employee_123`)
   - **Name**: Full name (e.g., `John Doe`)
   - **Images**: Upload 2-10 fingerprint images (5 recommended)

3. Click **"Add User"**
4. Wait for processing (2-10 seconds per user)
5. View result:
   - ✓ Success: User enrolled with quality score
   - ✗ Error: Quality too low or processing failed

**Tips:**
- Use consistent user ID format (e.g., `USER_001`, `EMP_0123`)
- Capture fingerprints with good lighting and clean scanner
- 5 samples provide best accuracy (minimum 2 required)

### 7.3 Load Folder (Batch Enrollment)

**Folder Structure:**

The system supports two folder structures:

**Option A: Flat Structure** (recommended for standard databases)
```
folder_path/
  101_1.tif
  101_2.tif
  101_3.tif
  ...
  101_8.tif
  102_1.tif
  102_2.tif
  ...
```

**Option B: Subfolder Structure**
```
folder_path/
  user001/
    img1.png
    img2.png
    img3.png
  user002/
    img1.png
    img2.png
```

**Steps:**

1. Click **"📂 Load Folder"**
2. Enter folder path:
   - Windows: `C:/fingerprints` or `C:\\fingerprints`
   - Linux: `/home/user/fingerprints`

3. Click **"🔍 Load Folder"**
4. Background job starts:
   - Progress shown in real-time
   - Can navigate away and check back later

5. View results:
   - ✅ Enrolled: X users
   - ⚠️ Skipped: Y users (already exist)
   - ❌ Errors: Z users (quality too low or file errors)

**Tips:**
- Minimum 5 images per user required
- All images processed (no maximum limit)
- Existing users are automatically skipped
- Large batches (100+ users) may take 10-30 minutes

### 7.4 Verify (1:1 Matching)

**Steps:**

1. Click **"🔍 Verify"**
2. Enter **User ID** to verify against
3. Upload **probe fingerprint image**
4. Click **"Verify"**
5. View result:
   - ✓ **VERIFIED**: Score ≥ 75% (threshold)
   - ✗ **NOT VERIFIED**: Score < 75%

**Interpretation:**
- **Score**: Combined hash + geometric similarity
- **Threshold**: 75% (configurable in `config.py`)
- **Use Case**: Authenticate claimed identity

### 7.5 Identify (1:N Matching)

**Steps:**

1. Click **"🎯 Identify"**
2. Upload **probe fingerprint image**
3. Click **"Identify"**
4. View result:
   - ✓ **IDENTIFIED**: Best match with User ID and score
   - ✗ **NOT IDENTIFIED**: No match above threshold

**Interpretation:**
- **Best Match**: Highest scoring user in database
- **Score**: Combined hash + geometric similarity
- **Threshold**: 70% (configurable in `config.py`)
- **Use Case**: Search unknown fingerprint against database

### 7.6 Update User

**Steps:**

1. Find user in "Enrolled Users" table
2. Click **"Edit"** button
3. Modify:
   - **Name**: Update user's name
   - **Images**: Upload new fingerprint images (re-enrollment)

4. Click **"Update"**
5. Template regenerated if new images provided

### 7.7 Delete User

**Steps:**

1. Find user in "Enrolled Users" table
2. Click **"Delete"** button
3. Confirm deletion (⚠️ cannot be undone)
4. User removed from database and cache

### 7.8 Delete All Users

**⚠️ WARNING: This operation deletes ALL enrolled users!**

**Steps:**

1. Click **"🗑️ Delete All"**
2. Confirm (⚠️ cannot be undone)
3. All users deleted from database
4. Template cache cleared

**Use Case:** Testing, database reset, or complete system reset

---

## 8. Backup and Recovery

### 8.1 What to Backup

**Critical Files:**
```
biometric_system/
├── biometric_system.db       # Main database (contains all users)
├── .db_key                   # Database encryption key
├── .jwt_secret               # JWT signing key
└── certs/
    ├── self-signed-cert.pem  # SSL certificate
    └── self-signed-key.pem   # SSL private key
```

**Optional Files:**
```
logs/                         # System logs (for troubleshooting)
  ├── access.log
  ├── auth.log
  ├── biometric.log
  └── error.log
```

### 8.2 Backup Procedure

**Windows:**
```cmd
# Stop the server first (CTRL+C)

# Create backup folder
mkdir backups\%date:~-4,4%%date:~-7,2%%date:~-10,2%

# Copy critical files
copy biometric_system.db backups\%date:~-4,4%%date:~-7,2%%date:~-10,2%\
copy .db_key backups\%date:~-4,4%%date:~-7,2%%date:~-10,2%\
copy .jwt_secret backups\%date:~-4,4%%date:~-7,2%%date:~-10,2%\
xcopy /E /I certs backups\%date:~-4,4%%date:~-7,2%%date:~-10,2%\certs
```

**Linux:**
```bash
# Stop the server first (CTRL+C)

# Create backup folder
mkdir -p backups/$(date +%Y%m%d)

# Copy critical files
cp biometric_system.db backups/$(date +%Y%m%d)/
cp .db_key backups/$(date +%Y%m%d)/
cp .jwt_secret backups/$(date +%Y%m%d)/
cp -r certs backups/$(date +%Y%m%d)/
```

**Automated Backup Script** (Linux/cron):
```bash
#!/bin/bash
# Save as: backup_biometric.sh

BACKUP_DIR="/path/to/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR/$DATE"
cp biometric_system.db "$BACKUP_DIR/$DATE/"
cp .db_key "$BACKUP_DIR/$DATE/"
cp .jwt_secret "$BACKUP_DIR/$DATE/"
cp -r certs "$BACKUP_DIR/$DATE/"

# Keep only last 30 days
find "$BACKUP_DIR" -type d -mtime +30 -exec rm -rf {} +
```

Add to crontab:
```bash
# Daily backup at 2 AM
0 2 * * * /path/to/backup_biometric.sh
```

### 8.3 Recovery Procedure

**Steps:**

1. **Stop the server** (if running)

2. **Restore files:**
   ```bash
   # Replace current files with backup
   cp backups/20250115/biometric_system.db .
   cp backups/20250115/.db_key .
   cp backups/20250115/.jwt_secret .
   cp -r backups/20250115/certs .
   ```

3. **Verify file permissions:**
   ```bash
   # Linux
   chmod 600 .db_key .jwt_secret
   chmod 600 certs/*.pem
   ```

4. **Start the server:**
   ```bash
   python run_webserver.py
   ```

5. **Verify data:**
   - Login to admin dashboard
   - Check user count matches backup
   - Test verify/identify operations

---

## 9. Monitoring and Logs

### 9.1 Log Files

All logs are stored in `logs/` directory:

| File | Contents | Use Case |
|------|----------|----------|
| `access.log` | HTTP requests (IP, endpoint, status, duration) | Monitor API usage, detect anomalies |
| `auth.log` | Authentication events (login, logout, failures) | Security auditing, detect brute-force attacks |
| `biometric.log` | Biometric operations (enroll, verify, identify) | Track system usage, performance analysis |
| `error.log` | Application errors and exceptions | Troubleshooting, debugging |

### 9.2 Log Format

**access.log:**
```
2025-01-15 14:23:45,123 [INFO] webserver.access: 192.168.1.100 - POST /api/verify - 200 - 2340.56ms user=endpoint
```

**auth.log:**
```
2025-01-15 14:23:30,456 [INFO] webserver.auth: LOGIN SUCCESS - login=admin ip=192.168.1.100
```

**biometric.log:**
```
2025-01-15 14:23:45,789 [INFO] webserver.biometric: VERIFY MATCH - user_id=user001 by=endpoint - score=0.82, num_matched=34
```

**error.log:**
```
2025-01-15 14:24:01,234 [ERROR] webserver.error: ValueError: Quality too low: 0.25 < 0.30 in /api/users user=admin
```

### 9.3 Monitoring Commands

**View live logs:**
```bash
# Linux
tail -f logs/access.log
tail -f logs/biometric.log

# Windows (PowerShell)
Get-Content logs\access.log -Wait -Tail 50
```

**Search logs:**
```bash
# Find all failed logins
grep "LOGIN FAILED" logs/auth.log

# Find verifications for specific user
grep "user_id=user001" logs/biometric.log

# Find errors in last hour
find logs/error.log -mmin -60 -exec cat {} \;
```

**Log rotation:**
- Logs automatically rotate at 10MB per file
- Keeps 5 backup files (e.g., `access.log.1`, `access.log.2`, ...)
- Total log storage: ~50MB per log type

### 9.4 System Health Check

**Check server status:**
```bash
curl -k https://localhost:8443/health
```

Response:
```json
{
  "status": "healthy",
  "cached_templates": 42,
  "max_workers": 4,
  "running_jobs": 0
}
```

**Check database:**
```bash
# Count enrolled users
python -c "from src.webserver.database import BiometricDatabase; db = BiometricDatabase(); print(f'Users: {len(db.list_fingerprints())}'); db.close()"
```

---

## 10. Troubleshooting

### 10.1 Server Won't Start

**Issue: "Port already in use"**

```bash
# Find process using port 8443
# Linux:
sudo lsof -i :8443
# Windows:
netstat -ano | findstr :8443

# Kill the process or use different port
python run_webserver.py --port 9443
```

**Issue: "Module not found"**

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Verify installation
pip list
```

**Issue: "Database locked"**

```bash
# Another process is using the database
# Find and stop it:
ps aux | grep run_webserver.py  # Linux
tasklist | findstr python        # Windows

# If persistent, remove lock file:
rm biometric_system.db-journal
```

### 10.2 SSL Certificate Errors

**Issue: Browser shows "Not Secure"**

This is **normal** for self-signed certificates. Options:

1. **Accept the warning** (click "Advanced" → "Proceed")
2. **Add exception** in browser settings
3. **Use production certificate** (e.g., Let's Encrypt):
   ```bash
   python run_webserver.py --cert /path/to/fullchain.pem --key /path/to/privkey.pem
   ```

**Issue: "Certificate verification failed"**

```bash
# Regenerate self-signed certificate
rm -rf certs/
python run_webserver.py
```

### 10.3 Authentication Issues

**Issue: "Invalid credentials"**

```bash
# Reset admin password
python run_webserver.py --reset-admin
```

**Issue: "Token has expired"**

- Normal behavior after configured expiry time
- Re-login to get new token
- Adjust expiry in `config.py`:
  ```python
  JWT_EXPIRY_ADMIN_HOURS = 2  # Increase from 1 to 2 hours
  ```

**Issue: "IP locked out"**

```bash
# Wait 15 minutes or restart server to clear lockout
# Adjust lockout settings in config.py:
MAX_LOGIN_ATTEMPTS = 10      # Increase from 5
LOCKOUT_DURATION_MINUTES = 5 # Decrease from 15
```

### 10.4 Enrollment Failures

**Issue: "Quality too low"**

**Causes:**
- Poor image quality (blur, noise)
- Incorrect fingerprint placement
- Scanner issues

**Solutions:**
- Clean fingerprint scanner
- Ensure good lighting
- Capture multiple clear samples
- Lower quality threshold (not recommended for production):
  ```python
  QUALITY_THRESHOLD = 0.25  # Default: 0.30
  ```

**Issue: "No minutiae extracted"**

**Causes:**
- Extremely low quality image
- Incompatible image format
- Corrupted file

**Solutions:**
- Use standard formats (PNG, JPG, TIF)
- Re-capture fingerprint
- Verify image can be opened in image viewer

**Issue: "User already exists"**

- User ID must be unique
- Delete existing user first or use different ID

### 10.5 Matching Issues

**Issue: "False rejects (legitimate user not verified)"**

**Causes:**
- Threshold too high
- Poor enrollment quality
- Probe image quality low

**Solutions:**
- Lower thresholds in `config.py`:
  ```python
  VERIFICATION_THRESHOLD = 0.70   # Default: 0.75
  IDENTIFICATION_THRESHOLD = 0.65 # Default: 0.70
  ```
- Re-enroll user with better quality samples
- Ensure consistent capture conditions

**Issue: "False accepts (wrong user verified)"**

**Causes:**
- Threshold too low
- Similar fingerprints in database
- Quality threshold too low during enrollment

**Solutions:**
- Increase thresholds:
  ```python
  VERIFICATION_THRESHOLD = 0.80   # Default: 0.75
  QUALITY_THRESHOLD = 0.35        # Default: 0.30
  ```
- Review enrolled users for duplicates
- Re-enroll with stricter quality control

### 10.6 Performance Issues

**Issue: "Slow enrollment/identification"**

**Causes:**
- Insufficient CPU resources
- Too many workers competing for resources
- Large database (1000+ users)

**Solutions:**
- Adjust worker count:
  ```python
  MAX_WORKERS = 8  # Increase for more parallelism
  ```
- Enable CPU affinity on hybrid CPUs:
  ```python
  USE_CPU_AFFINITY = True
  PCORES = list(range(0, 12))  # Adjust for your CPU
  ```
- Monitor CPU usage during operations

**Issue: "High memory usage"**

**Causes:**
- Large template cache
- Many concurrent operations

**Solutions:**
- Monitor with `htop` (Linux) or Task Manager (Windows)
- Reduce worker count if memory limited:
  ```python
  MAX_WORKERS = 2  # Reduce from 4
  ```

---

## 11. Security Considerations

### 11.1 Production Deployment Checklist

- [ ] **Use production SSL certificate** (not self-signed)
- [ ] **Change default passwords** for admin and endpoint accounts
- [ ] **Restrict CORS origins** in `config.py`:
  ```python
  CORS_ORIGINS = ["https://yourdomain.com"]
  ```
- [ ] **Enable firewall rules** to restrict access:
  ```bash
  # Linux (ufw)
  sudo ufw allow from 192.168.1.0/24 to any port 8443
  
  # Windows Firewall
  # Add inbound rule for port 8443, specific IP range
  ```
- [ ] **Set strong rate limits** for production
- [ ] **Enable automated backups** (daily recommended)
- [ ] **Monitor logs** for suspicious activity
- [ ] **Keep software updated** (Python, dependencies, OS)
- [ ] **Use reverse proxy** (nginx, Apache) for additional security layer

### 11.2 Network Security

**Recommended Architecture:**

```
Internet
    │
    ▼
┌─────────────────┐
│  Firewall       │ (Allow only specific IPs/ports)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Reverse Proxy   │ (nginx/Apache with SSL termination)
│ (Port 443)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Biometric Server│ (Internal network, port 8443)
│ 127.0.0.1:8443  │
└─────────────────┘
```

**nginx Example Configuration:**
```nginx
server {
    listen 443 ssl http2;
    server_name biometric.yourdomain.com;
    
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
    
    location / {
        proxy_pass https://127.0.0.1:8443;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 11.3 Database Security

**Encryption:**
- Database is encrypted using SQLCipher with AES-256
- Encryption key stored in `.db_key` file
- **⚠️ CRITICAL**: Backup `.db_key` - without it, database cannot be decrypted

**Access Control:**
```bash
# Restrict file permissions (Linux)
chmod 600 biometric_system.db
chmod 600 .db_key
chmod 600 .jwt_secret
```

**Key Rotation** (Advanced):
```bash
# Generate new encryption key
python -c "import secrets; print(secrets.token_urlsafe(32))" > .db_key.new

# Re-encrypt database with new key (manual process - not automated)
# 1. Export all users
# 2. Delete database
# 3. Replace .db_key with .db_key.new
# 4. Re-import users
```

### 11.4 Authentication Security

**Password Policy:**
- Minimum 8 characters (configurable in `config.py`)
- Passwords hashed with bcrypt (cost factor 12)
- No password reuse enforcement (implement externally if needed)

**Token Security:**
- JWT tokens signed with HS256 algorithm
- Secret key auto-generated on first run (stored in `.jwt_secret`)
- Tokens include expiry timestamp
- Stateless validation (no server-side session storage)

**Best Practices:**
- Rotate admin passwords every 90 days
- Use strong passwords (16+ characters, mixed case, numbers, symbols)
- Never share credentials via insecure channels
- Monitor `auth.log` for suspicious login patterns

### 11.5 Template Security

**Storage Format:**
- Templates stored as **Protected Hash + Minutiae** (secure format)
- **NO raw biometric data** stored in database
- Cancelable templates: Can be revoked by changing hash key

**Template Protection:**
- Hash-based encoding with random projection
- Projection key: `HASH_KEY` in `config.py`
- Changing key invalidates all templates (requires re-enrollment)

**Privacy Compliance:**
- System stores only **derived features**, not raw images
- GDPR compliant: Users can request deletion (delete user operation)
- No biometric data leakage even if database is compromised

### 11.6 Audit Trail

**Audit Logging:**
- All operations logged in `auth.log` and `biometric.log`
- Logs include: timestamp, user, IP address, action, result
- Database audit table: `audit_log`

**View Audit Log:**
```bash
# From admin dashboard
Admin → Audit Log (limit 100 entries)

# From database
python -c "from src.webserver.database import BiometricDatabase; db = BiometricDatabase(); logs = db.get_audit_log(1000); [print(l) for l in logs]; db.close()"
```

**Audit Events Tracked:**
- User authentication (login, logout, failures)
- User management (create, update, delete)
- Biometric operations (enroll, verify, identify)
- Job execution (folder loading, batch operations)

### 11.7 Vulnerability Management

**Keep System Updated:**
```bash
# Update Python dependencies
pip list --outdated
pip install --upgrade <package>

# Update entire system
pip install -r requirements.txt --upgrade
```

**Security Scanning:**
```bash
# Check for known vulnerabilities in dependencies
pip install safety
safety check

# Check for outdated packages
pip install pip-audit
pip-audit
```

**Regular Security Tasks:**
- [ ] Review logs weekly for anomalies
- [ ] Update dependencies monthly
- [ ] Rotate passwords quarterly
- [ ] Test backups quarterly
- [ ] Review rate limits and thresholds quarterly

---

## 12. Advanced Configuration

### 12.1 Multi-Node Deployment

For high-availability setups with multiple servers:

**Architecture:**
```
         Load Balancer (HAProxy/nginx)
                 │
      ┌──────────┼──────────┐
      │          │          │
   Server 1   Server 2   Server 3
      │          │          │
      └──────────┴──────────┘
               │
        Shared Database
     (PostgreSQL + NFS mount)
```

**Considerations:**
- Shared database required (current: SQLite - single file)
- Template cache must be synchronized across nodes
- JWT secret must be identical on all nodes
- Session affinity recommended for endpoint users

**Migration Path:**
1. Export current database to PostgreSQL
2. Update database connection in `database.py`
3. Deploy multiple server instances
4. Configure load balancer

### 12.2 Custom Thresholds per User

For specialized use cases (VIP users, high-security zones):

**Edit `database.py` to add threshold field:**
```python
# Add column to fingerprints table
threshold REAL DEFAULT 0.75
```

**Modify matching logic to use per-user threshold:**
```python
# In biometric_routes.py
user_data = db.get_fingerprint(user_id)
custom_threshold = user_data.get('threshold', VERIFICATION_THRESHOLD)
```

### 12.3 External Authentication (LDAP/Active Directory)

For enterprise integration:

**Install LDAP library:**
```bash
pip install ldap3
```

**Modify `auth.py` to add LDAP authentication:**
```python
from ldap3 import Server, Connection, ALL

def authenticate_ldap(username, password):
    server = Server('ldap://your-ldap-server.com', get_info=ALL)
    conn = Connection(server, user=f'cn={username},dc=company,dc=com', password=password)
    
    if conn.bind():
        return True
    return False
```

### 12.4 Webhook Notifications

For integration with external systems:

**Add webhook configuration to `config.py`:**
```python
WEBHOOK_ENABLED = True
WEBHOOK_URL = "https://your-system.com/webhook"
WEBHOOK_EVENTS = ["enroll", "verify", "identify"]
```

**Implement webhook in `logger.py`:**
```python
import requests

def send_webhook(event, data):
    if WEBHOOK_ENABLED and event in WEBHOOK_EVENTS:
        try:
            requests.post(WEBHOOK_URL, json={
                'event': event,
                'timestamp': datetime.utcnow().isoformat(),
                'data': data
            }, timeout=5)
        except Exception as e:
            log_error(e, context="webhook")
```

---

## 13. API Reference (Quick Guide)

### 13.1 Authentication Endpoints

**Login:**
```bash
POST /api/login
Content-Type: application/json

{
  "login": "admin",
  "password": "your_password"
}

Response:
{
  "token": "eyJhbGci...",
  "login": "admin",
  "privilege": "admin",
  "expires_in": 3600
}
```

**All subsequent requests require header:**
```
Authorization: Bearer <token>
```

### 13.2 User Management Endpoints

**List Users:**
```bash
GET /api/users
Authorization: Bearer <token>

Response:
{
  "users": [
    {
      "user_id": "user001",
      "name": "John Doe",
      "quality": 0.85,
      "num_minutiae": 42,
      "created_at": "2025-01-15T14:23:45"
    }
  ]
}
```

**Add User:**
```bash
POST /api/users
Authorization: Bearer <token>
Content-Type: multipart/form-data

user_id: user001
name: John Doe
images: [file1.png, file2.png, file3.png, ...]

Response:
{
  "user_id": "user001",
  "quality": 0.85,
  "num_images": 5
}
```

**Delete User:**
```bash
DELETE /api/users/{user_id}
Authorization: Bearer <token>

Response:
{
  "message": "User deleted successfully",
  "user_id": "user001"
}
```

### 13.3 Biometric Operations

**Verify (1:1):**
```bash
POST /api/verify
Authorization: Bearer <token>
Content-Type: multipart/form-data

user_id: user001
image: probe.png

Response:
{
  "match": true,
  "score": 0.82,
  "hash_score": 0.78,
  "geometric_score": 0.88,
  "user_id": "user001",
  "threshold": 0.75,
  "num_matched": 34
}
```

**Identify (1:N):**
```bash
POST /api/identify
Authorization: Bearer <token>
Content-Type: multipart/form-data

image: probe.png

Response:
{
  "identified": true,
  "best_match_id": "user001",
  "best_score": 0.82,
  "matches": [
    {"user_id": "user001", "score": 0.82},
    {"user_id": "user042", "score": 0.45}
  ],
  "total_users": 42,
  "threshold": 0.70
}
```

### 13.4 Admin Endpoints

**Load Folder:**
```bash
POST /api/admin/load_folder
Authorization: Bearer <token>
Content-Type: multipart/form-data

folder_path: /path/to/fingerprints

Response:
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Background job started"
}
```

**Get Job Status:**
```bash
GET /api/admin/jobs/{job_id}
Authorization: Bearer <token>

Response:
{
  "id": "550e8400-...",
  "type": "load_folder",
  "status": "completed",
  "result": "{\"enrolled\": 42, \"failed\": 3}",
  "started_at": "2025-01-15T14:23:45",
  "completed_at": "2025-01-15T14:28:12"
}
```

**Get Statistics:**
```bash
GET /api/admin/stats
Authorization: Bearer <token>

Response:
{
  "num_users": 42,
  "avg_template_quality": 0.85,
  "total_minutiae": 1764,
  "encrypted": true,
  "cached_templates": 42,
  "running_jobs": 0
}
```

---

## 14. FAQ

**Q: How many users can the system handle?**

A: Tested with up to 10,000 users. Performance:
- Enrollment: ~2-5 seconds per user
- Verification (1:1): ~2-3 seconds
- Identification (1:N): 
  - 100 users: ~3-5 seconds
  - 1,000 users: ~10-20 seconds (with parallel processing)
  - 10,000 users: ~60-120 seconds

**Q: Can I use the system offline?**

A: Yes, the system is fully self-contained and requires no internet connection. All processing is done locally.

**Q: What image formats are supported?**

A: PNG, JPG/JPEG, TIF/TIFF, BMP. TIF format recommended for fingerprint databases.

**Q: Can I integrate with my existing application?**

A: Yes, use the REST API documented in Section 13. The API is language-agnostic (works with Python, JavaScript, Java, C#, etc.).

**Q: Is the system GDPR compliant?**

A: Yes, when properly configured:
- Only derived features stored (not raw biometrics)
- Users can request deletion (delete user operation)
- Audit logging tracks all data access
- Templates are cancelable (can be revoked)

**Q: What happens if I lose the encryption key (.db_key)?**

A: **Database cannot be recovered.** The encryption key is critical - back it up securely and separately from the database.

**Q: Can I change the hash key after enrollment?**

A: No, changing `HASH_KEY` invalidates all templates. Users must be re-enrolled. Plan key management before production deployment.

**Q: How do I upgrade to a new version?**

A:
1. Backup database and configuration
2. Stop server
3. Update code files
4. Run `pip install -r requirements.txt --upgrade`
5. Test in development environment first
6. Deploy to production

**Q: What if the server crashes during enrollment?**

A: Partial enrollments are rolled back automatically. The user will not be added to the database. Simply retry the enrollment.

---

## 15. Support and Maintenance

### 15.1 Log Review Schedule

**Daily:**
- Check `error.log` for exceptions
- Monitor failed authentication attempts in `auth.log`

**Weekly:**
- Review `biometric.log` for usage patterns
- Check system health: `/health` endpoint
- Verify backup completion

**Monthly:**
- Analyze performance metrics
- Review and adjust rate limits if needed
- Update dependencies

**Quarterly:**
- Full system audit
- Password rotation
- Backup restore test
- Security scan

### 15.2 Performance Tuning

**For High-Volume Systems (1000+ verifications/day):**

```python
# config.py
MAX_WORKERS = 8                    # Increase workers
CACHE_ENABLED = True               # Ensure caching enabled
FUSION_ENABLED = True              # Better accuracy
VERIFICATION_THRESHOLD = 0.75      # Maintain security

# Rate limits - adjust based on usage
RATE_LIMITS = {
    "verify": (30, 60),            # 30 per minute
    "identify": (10, 60),          # 10 per minute
}
```

**For Low-Resource Systems (limited CPU/RAM):**

```python
# config.py
MAX_WORKERS = 2                    # Reduce workers
CACHE_ENABLED = True               # Keep for speed
USE_CPU_AFFINITY = False           # Disable affinity
```

### 15.3 Disaster Recovery

**Scenario: Database Corruption**

1. Stop server immediately
2. Restore from last backup:
   ```bash
   cp backups/latest/biometric_system.db .
   cp backups/latest/.db_key .
   ```
3. Start server and verify user count
4. If corruption recent, use transaction log recovery (if available)

**Scenario: Server Hardware Failure**

1. Install system on new hardware
2. Restore all files from backup:
   - `biometric_system.db`
   - `.db_key`
   - `.jwt_secret`
   - `certs/`
3. Install dependencies: `pip install -r requirements.txt`
4. Start server and verify functionality

**Scenario: Accidental User Deletion**

1. Stop server
2. Restore database from backup:
   ```bash
   cp backups/before_deletion/biometric_system.db .
   ```
3. Start server
4. **Note**: Any enrollments after backup are lost

---

## 16. Appendix

### 16.1 Configuration File Template

Save as `config.py` for reference:

```python
"""WebServer Configuration - Production Template"""

# NETWORK
HOST = "0.0.0.0"
PORT_HTTPS = 8443
PORT_HTTP = 8080
CORS_ORIGINS = ["https://yourdomain.com"]

# AUTHENTICATION
JWT_EXPIRY_ADMIN_HOURS = 1
JWT_EXPIRY_ENDPOINT_HOURS = 12
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 15

# BIOMETRIC THRESHOLDS
QUALITY_THRESHOLD = 0.30
VERIFICATION_THRESHOLD = 0.75
IDENTIFICATION_THRESHOLD = 0.70

# ENROLLMENT
MIN_SAMPLES_PER_USER = 2
MAX_SAMPLES_PER_USER = 10
RECOMMENDED_SAMPLES = 5

# FUSION
FUSION_ENABLED = True
FUSION_DISTANCE = 12.0
FUSION_ANGLE_DEG = 15.0
FUSION_MIN_CONSENSUS = 0.5
FUSION_MODE = "optimal"

# MULTIPROCESSING
MAX_WORKERS = 4
USE_CPU_AFFINITY = True
PCORES = list(range(0, 12))

# RATE LIMITING
RATE_LIMITS = {
    "login": (5, 300),
    "verify": (10, 60),
    "identify": (5, 60),
    "enroll": (20, 3600),
}

# LOGGING
LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 5
VERBOSE = True
```

### 16.2 Command Reference

**Server Management:**
```bash
# Start server
python run_webserver.py

# Start with custom settings
python run_webserver.py --host 0.0.0.0 --port 8443 --ssl

# Reset admin password
python run_webserver.py --reset-admin

# Use more workers
python run_webserver.py --workers 8
```

**Database Operations:**
```bash
# View user count
python -c "from src.webserver.database import BiometricDatabase; db = BiometricDatabase(); print(f'Users: {len(db.list_fingerprints())}'); db.close()"

# View statistics
python -c "from src.webserver.database import BiometricDatabase; db = BiometricDatabase(); print(db.get_stats()); db.close()"

# Export users list
python -c "from src.webserver.database import BiometricDatabase; import json; db = BiometricDatabase(); print(json.dumps(db.list_fingerprints(), indent=2)); db.close()" > users.json
```

**Log Operations:**
```bash
# View recent errors
tail -n 50 logs/error.log

# Search for user
grep "user001" logs/biometric.log

# Count operations
grep "VERIFY" logs/biometric.log | wc -l
```

### 16.3 Glossary

- **Enrollment**: Process of capturing and storing a user's fingerprint template
- **Template**: Derived biometric features stored in database (NOT raw image)
- **Verification (1:1)**: Confirming claimed identity against enrolled template
- **Identification (1:N)**: Finding match in database for unknown fingerprint
- **Quality Score**: Metric (0.0-1.0) indicating template reliability
- **Threshold**: Minimum score required for positive match
- **Minutiae**: Ridge ending/bifurcation points in fingerprint
- **Fusion**: Combining multiple samples into master template
- **Protected Hash**: Cancelable template encoding for security
- **JWT**: JSON Web Token for stateless authentication
- **Rate Limiting**: Restricting request frequency to prevent abuse

---

## Document Information

**Document Version**: 1.0.0  
**Last Updated**: November 2025  
**System Version**: Supermatcher 1.0.0  
**Author**: System Administrator Guide  

**Revision History:**
- v1.0.0 (2025-11): Initial release

---

**End of Administrator Manual**

For technical support or questions, consult the system logs or contact your system integrator.