"""
Run Webserver - Startup Script
Interactive first-time setup and server launcher.
"""

import os
import sys
from pathlib import Path
import getpass
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.webserver.config import DB_PATH, PRIVILEGE_ADMIN, PRIVILEGE_ENDPOINT, HOST, PORT_HTTPS, PORT_HTTP
from src.webserver.database import BiometricDatabase


def first_time_setup():
    """Interactive setup for first run."""
    print("=" * 70)
    print("🔐 BIOMETRIC WEBSERVER - FIRST-TIME SETUP")
    print("=" * 70)
    print()
    
    # Admin account
    print("Create ADMIN account:")
    admin_user = input("  Username [admin]: ").strip() or "admin"
    
    while True:
        admin_pass = getpass.getpass("  Password (min 8 chars): ")
        if len(admin_pass) < 8:
            print("  ⚠ Password too short!")
            continue
        admin_confirm = getpass.getpass("  Confirm password: ")
        if admin_pass != admin_confirm:
            print("  ⚠ Passwords don't match!")
            continue
        break
    
    # Endpoint account
    print()
    print("Create ENDPOINT account:")
    endpoint_user = input("  Username [endpoint]: ").strip() or "endpoint"
    
    while True:
        endpoint_pass = getpass.getpass("  Password (min 8 chars): ")
        if len(endpoint_pass) < 8:
            print("  ⚠ Password too short!")
            continue
        endpoint_confirm = getpass.getpass("  Confirm password: ")
        if endpoint_pass != endpoint_confirm:
            print("  ⚠ Passwords don't match!")
            continue
        break
    
    # SSL
    print()
    use_ssl = input("Use HTTPS with SSL? [Y/n]: ").lower() != 'n'
    
    port = PORT_HTTPS if use_ssl else PORT_HTTP
    port_input = input(f"  Port [{port}]: ").strip()
    if port_input:
        try:
            port = int(port_input)
        except ValueError:
            print(f"  ⚠ Invalid port, using default: {port}")
    
    # Create database and users
    print()
    print("Creating database...")
    db = BiometricDatabase()
    
    db.create_auth_user(admin_user, admin_pass, PRIVILEGE_ADMIN)
    print(f"✓ Admin user '{admin_user}' created")
    
    db.create_auth_user(endpoint_user, endpoint_pass, PRIVILEGE_ENDPOINT)
    print(f"✓ Endpoint user '{endpoint_user}' created")
    
    db.close()
    
    print()
    print("=" * 70)
    print("✅ SETUP COMPLETE!")
    print("=" * 70)
    print()
    
    return use_ssl, port


def main():
    """Start server."""
    parser = argparse.ArgumentParser(description="Biometric WebServer")
    parser.add_argument("--host", default=HOST, help="Server host")
    parser.add_argument("--port", type=int, help="Server port")
    parser.add_argument("--ssl", action="store_true", help="Enable HTTPS/TLS")
    parser.add_argument("--cert", help="SSL certificate file")
    parser.add_argument("--key", help="SSL key file")
    parser.add_argument("--reset-admin", action="store_true", help="Reset admin password")
    parser.add_argument("--workers", type=int, help="Number of uvicorn workers")
    
    args = parser.parse_args()
    
    # Check if first run (check for auth users, not just DB existence)
    needs_setup = False
    if not args.reset_admin:
        try:
            db = BiometricDatabase()
            stats = db.get_stats()
            db.close()
            # Setup needed if no auth users exist
            if stats.get('num_auth_users', 0) == 0:
                needs_setup = True
        except Exception:
            needs_setup = True
    
    if needs_setup:
        use_ssl, port = first_time_setup()
        args.ssl = use_ssl
        if not args.port:
            args.port = port
    
    # Reset admin password
    if args.reset_admin:
        print("🔄 RESET ADMIN PASSWORD")
        print()
        
        new_pass = getpass.getpass("New admin password (min 8 chars): ")
        if len(new_pass) < 8:
            print("⚠ Password too short!")
            return
        
        confirm = getpass.getpass("Confirm password: ")
        if new_pass != confirm:
            print("⚠ Passwords don't match!")
            return
        
        db = BiometricDatabase()
        if db.reset_admin_password(new_pass):
            print("✓ Admin password reset successfully!")
        else:
            print("✗ Failed to reset password!")
        db.close()
        return
    
    # Determine port
    port = args.port or (PORT_HTTPS if args.ssl else PORT_HTTP)
    
    # Print startup info
    print()
    print("=" * 70)
    print("STARTING BIOMETRIC WEBSERVER")
    print("=" * 70)
    print()
    
    # SSL configuration
    ssl_keyfile = None
    ssl_certfile = None
    
    if args.ssl:
        if args.cert and args.key:
            ssl_certfile = args.cert
            ssl_keyfile = args.key
            print(f"🔒 HTTPS enabled with custom certificate")
        else:
            # Generate self-signed certificate
            print("⚠ No SSL certificate provided, generating self-signed certificate...")
            try:
                ssl_certfile, ssl_keyfile = generate_self_signed_cert()
                print(f"✓ Self-signed certificate generated")
                print(f"  ⚠ Browser will show security warning (normal for self-signed certs)")
            except Exception as e:
                print(f"✗ Failed to generate certificate: {e}")
                print("  Falling back to HTTP...")
                args.ssl = False
    
    # Start server with uvicorn
    import uvicorn
    
    print(f"\n🚀 Server starting on {args.host}:{port}")
    print(f"   Protocol: {'HTTPS' if args.ssl else 'HTTP'}")
    print(f"   Workers: {args.workers or 1}")
    print()
    print(f"📱 Web Interface: {'https' if args.ssl else 'http'}://localhost:{port}/")
    print(f"📚 API Docs: {'https' if args.ssl else 'http'}://localhost:{port}/docs")
    print()
    print("Press CTRL+C to stop")
    print("=" * 70)
    print()
    
    try:
        uvicorn.run(
            "src.webserver.server:app",
            host=args.host,
            port=port,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            reload=False,
            workers=args.workers or 1,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\nShutting down...")


def generate_self_signed_cert():
    """
    Generate self-signed SSL certificate.
    
    Returns:
        Tuple of (certfile, keyfile) paths
    """
    from datetime import datetime, timedelta, timezone
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    
    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    
    # Create self-signed certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, u"PT"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"Portugal"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, u"Localhost"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"Biometric System"),
        x509.NameAttribute(NameOID.COMMON_NAME, u"localhost"),
    ])
    
    now = datetime.now(timezone.utc)
    
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        private_key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        now
    ).not_valid_after(
        now + timedelta(days=365)
    ).add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName(u"localhost"),
            x509.DNSName(u"127.0.0.1"),
        ]),
        critical=False,
    ).sign(private_key, hashes.SHA256())
    
    # Save to files
    from src.webserver.config import SSL_CERT_DIR
    SSL_CERT_DIR.mkdir(exist_ok=True, parents=True)
    
    certfile = str(SSL_CERT_DIR / "self-signed-cert.pem")
    keyfile = str(SSL_CERT_DIR / "self-signed-key.pem")
    
    # Write certificate
    with open(certfile, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    
    # Write private key
    with open(keyfile, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))
    
    return certfile, keyfile


if __name__ == "__main__":
    main()
