# 🔐 Biometric Fingerprint Authentication System

A production-ready, secure fingerprint identification and verification system with web interface, built on the **Supermatcher v1.0** hybrid fingerprint matching pipeline.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🌟 Features

### Core Biometric Capabilities
- **1:N Identification**: Match a fingerprint against all enrolled users
- **1:1 Verification**: Verify identity against a specific user's template
- **Multi-sample Fusion**: Combine multiple fingerprint samples for robust enrollment
- **Cancelable Templates**: Protected biometric templates using random projection (non-invertible, revocable)
- **Quality Assessment**: Automatic quality evaluation with adaptive thresholds

### Security & Authentication
- **Encrypted Database**: SQLCipher-based encrypted storage
- **JWT Authentication**: Secure token-based authentication
- **Role-Based Access Control**: Admin and Endpoint privilege levels
- **Rate Limiting**: Protection against brute-force attacks
- **HTTPS/TLS Support**: Secure communication (with HTTP fallback)
- **Audit Logging**: Comprehensive operation tracking

### Architecture
- **RESTful API**: FastAPI-based REST endpoints with Swagger UI
- **Async Processing**: Background job queue for long-running operations
- **ProcessPool Workers**: CPU-bound tasks distributed across worker processes
- **Template Caching**: In-memory template cache for fast matching
- **Web Interface**: Modern HTML/CSS/JS frontend for management

## 🏗️ Architecture Overview

### Supermatcher v1.0 Pipeline

The system implements a modular fingerprint processing pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                    FINGERPRINT PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. PREPROCESSING                                               │
│     ├─ Load & Normalize                                         │
│     ├─ Segmentation (block variance)                            │
│     └─ Coherence Diffusion                                      │
│                                                                 │
│  2. FEATURE EXTRACTION                                          │
│     ├─ Orientation & Frequency Estimation                       │
│     ├─ Log-Gabor Enhancement                                    │
│     ├─ Binarization & Thinning                                  │
│     ├─ Minutiae Detection & Validation                          │
│     └─ Pore Detection (Level-3, optional)                       │
│                                                                 │
│  3. TEMPLATE CREATION                                           │
│     ├─ Feature Vector Construction (736D)                       │
│     ├─ Quality Assessment                                       │
│     └─ Protected Hash Generation (Random Projection)            │
│                                                                 │
│  4. MATCHING                                                    │
│     ├─ Stage 1: Hash-based filtering (fast)                     │
│     └─ Stage 2: Geometric reranking (RANSAC-based)              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📋 Requirements

- **Python 3.8+**
- **Dependencies**: See [requirements.txt](requirements.txt)
  - FastAPI & Uvicorn (web server)
  - OpenCV & NumPy (image processing)
  - SciPy (scientific computing)
  - SQLCipher (encrypted database)
  - bcrypt & PyJWT (authentication)
  - psutil (system monitoring)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/biometric-system.git
cd biometric-system

# Install dependencies
pip install -r requirements.txt
```

### 2. First-Time Setup

Run the interactive setup to create admin and endpoint accounts:

```bash
python run_webserver.py
```

The setup wizard will:
- Create admin account (full system access)
- Create endpoint account (biometric operations only)
- Generate SSL certificates (self-signed for development)
- Initialize encrypted database
- Set up logging directories

### 3. Start the Server

After setup, the server starts automatically. For subsequent runs:

```bash
python run_webserver.py
```

**Server URLs**:
- **HTTPS**: `https://localhost:8443`
- **HTTP** (fallback): `http://localhost:8080`
- **API Docs**: `https://localhost:8443/docs`


## ⚙️ Configuration

Edit `src/webserver/config.py` to customize:


---

**⚠️ Important**: This system is designed for development and testing purposes. For production deployment, ensure proper security hardening, use production-grade SSL certificates, and follow your organization's security policies.
