"""Database Module - SQLite Encrypted Database Management

Handles all database operations for the biometric webserver including:
- User authentication (admin/endpoint)
- Fingerprint template storage (SECURE: Protected hash + Minutiae only)
- Audit logging
- Background jobs tracking

Tables:
- auth_users: Authentication credentials
- fingerprints: Biometric templates (secure format, NO pickle)
- audit_log: All operations log
- jobs: Background job status

Security: Templates stored as Protected hash + Minutiae JSON (no raw features)

"""

from __future__ import annotations

import hashlib
from sqlcipher3 import dbapi2 as sqlite
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bcrypt

from .config import (
    DB_PATH,
    DB_ENCRYPTION_KEY,
    PRIVILEGE_ADMIN,
    PRIVILEGE_ENDPOINT,
)


# ============================================================================
# DATABASE CLASS
# ============================================================================

class BiometricDatabase:
    """SQLite database for biometric system with encryption support."""
    
    def __init__(self, db_path: Path = None, encryption_key: str = None):
        """Initialize database connection.
        
        Args:
            db_path: Path to database file
            encryption_key: Encryption key for SQLCipher
        """
        self.db_path = db_path or DB_PATH
        self.encryption_key = encryption_key or DB_ENCRYPTION_KEY
        self.conn = None
        self.encrypted = False
        
        # Try to connect
        self._connect()
        
        # Create tables if needed
        self._create_tables()
    
    def _connect(self):
        """Connect to database with encryption if available."""
        # Try SQLCipher first
        try:
            self.conn = sqlite.connect(str(self.db_path))
            self.conn.execute(f"PRAGMA key = '{self.encryption_key}'")
            self.conn.execute("PRAGMA cipher_compatibility = 3")
            # Test if encryption works
            self.conn.execute("SELECT count(*) FROM sqlite_master")
            self.encrypted = True
            print("✓ Using encrypted database (SQLCipher)")
        except (ImportError, Exception) as e:
            # Fallback to standard SQLite
            print(f"⚠ SQLCipher not available, using standard SQLite: {e}")
            self.conn = sqlite.connect(str(self.db_path))
            self.encrypted = False
        
        # Enable foreign keys
        self.conn.execute("PRAGMA foreign_keys = ON")
        
        # Row factory for dict-like access
        self.conn.row_factory = sqlite.Row
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Auth users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS auth_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                privilege TEXT NOT NULL CHECK(privilege IN ('admin', 'endpoint')),
                failed_attempts INTEGER DEFAULT 0,
                locked_until TIMESTAMP NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP NULL
            )
        """)
        
        # Fingerprints table - SECURE VERSION (Protected + Minutiae only)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fingerprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                protected_hash BLOB NOT NULL,
                bit_length INTEGER NOT NULL,
                minutiae_json TEXT,
                quality REAL NOT NULL,
                num_minutiae INTEGER,
                num_samples INTEGER,
                fused BOOLEAN DEFAULT 0,
                source_count INTEGER DEFAULT 1,
                consensus_score REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                template_blob BLOB
            )
        """)
        
        # Audit log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                username TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT,
                result TEXT,
                ip_address TEXT
            )
        """)
        
        # Background jobs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                status TEXT NOT NULL,
                progress TEXT,
                result TEXT,
                error TEXT,
                created_by TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fingerprints_user_id ON fingerprints(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
        
        self.conn.commit()
    
    # ========================================================================
    # AUTHENTICATION MANAGEMENT
    # ========================================================================
    
    def create_auth_user(self, username: str, password: str, privilege: str) -> bool:
        """Create authentication user.
        
        Args:
            username: Username
            password: Plain text password (will be hashed)
            privilege: 'admin' or 'endpoint'
            
        Returns:
            True if created successfully
        """
        try:
            # Hash password
            password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
            
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO auth_users (username, password_hash, privilege) VALUES (?, ?, ?)",
                (username, password_hash, privilege)
            )
            self.conn.commit()
            
            self._log_audit(username, "USER_CREATED", f"Created {privilege} user", "success")
            return True
        
        except sqlite.IntegrityError:
            return False
    
    def authenticate(self, username: str, password: str) -> Optional[Dict[str, str]]:
        """Authenticate user.
        
        Args:
            username: Username
            password: Plain text password
            
        Returns:
            Dict with user info or None if authentication fails
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM auth_users WHERE username = ?",
            (username,)
        )
        
        row = cursor.fetchone()
        if not row:
            self._log_audit(username, "LOGIN_FAILED", "User not found", "failure")
            return None
        
        user = dict(row)
        
        # Check if locked
        if user["locked_until"]:
            locked_until = datetime.fromisoformat(user["locked_until"])
            if datetime.now(timezone.utc) < locked_until:
                self._log_audit(username, "LOGIN_FAILED", "Account locked", "failure")
                return None
            else:
                # Unlock account
                cursor.execute(
                    "UPDATE auth_users SET locked_until = NULL, failed_attempts = 0 WHERE username = ?",
                    (username,)
                )
                self.conn.commit()
        
        # Verify password
        if bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
            # Reset failed attempts
            cursor.execute(
                "UPDATE auth_users SET failed_attempts = 0, last_login = ? WHERE username = ?",
                (datetime.now(timezone.utc).isoformat(), username)
            )
            self.conn.commit()
            
            self._log_audit(username, "LOGIN_SUCCESS", "", "success")
            
            return {
                "login": user["username"],
                "privilege": user["privilege"]
            }
        else:
            # Increment failed attempts
            failed_attempts = user["failed_attempts"] + 1
            cursor.execute(
                "UPDATE auth_users SET failed_attempts = ? WHERE username = ?",
                (failed_attempts, username)
            )
            
            # Lock account if too many attempts
            if failed_attempts >= 5:
                from datetime import timedelta
                lock_until = datetime.now(timezone.utc) + timedelta(minutes=15)
                cursor.execute(
                    "UPDATE auth_users SET locked_until = ? WHERE username = ?",
                    (lock_until.isoformat(), username)
                )
                self._log_audit(username, "ACCOUNT_LOCKED", f"Too many failed attempts ({failed_attempts})", "failure")
            
            self.conn.commit()
            self._log_audit(username, "LOGIN_FAILED", "Invalid password", "failure")
            return None
    
    def reset_admin_password(self, new_password: str) -> bool:
        """Reset admin password (for recovery).
        
        Args:
            new_password: New plain text password
            
        Returns:
            True if reset successfully
        """
        try:
            password_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
            
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE auth_users SET password_hash = ?, failed_attempts = 0, locked_until = NULL WHERE privilege = 'admin'",
                (password_hash,)
            )
            self.conn.commit()
            
            self._log_audit("system", "ADMIN_PASSWORD_RESET", "", "success")
            return True
        except Exception:
            return False
    
    # ========================================================================
    # FINGERPRINT MANAGEMENT
    # ========================================================================
    
    def add_fingerprint(
        self,
        user_id: str,
        name: str,
        template_obj: Any,
        username: str = "system"
    ) -> bool:
        """Add fingerprint template to database.
        
        Args:
            user_id: Unique user identifier
            name: User's name
            template_obj: FingerprintTemplate object from supermatcher v1.0
            username: Who performed the operation
            
        Returns:
            True if added successfully
        """
        try:
            # Serialize template SECURELY (Protected + Minutiae only)
            from src.models_serialization import template_to_secure_dict
            import json
            
            secure_data = template_to_secure_dict(template_obj)
            
            # Extract components
            protected_hash = secure_data['protected']
            bit_length = secure_data['bit_length']
            minutiae_json = json.dumps(secure_data['minutiae'])
            quality = secure_data['quality']
            num_minutiae = len(secure_data['minutiae'])
            num_samples = secure_data['source_count']
            fused = secure_data['fused']
            source_count = secure_data['source_count']
            consensus_score = secure_data['consensus_score']
            
            # SECURITY: No pickle blob saved (only Protected + Minutiae)
            # template_blob column kept NULL for backward compatibility
            
            cursor = self.conn.cursor()
            cursor.execute(
                """INSERT INTO fingerprints 
                   (user_id, name, protected_hash, bit_length, minutiae_json, 
                    quality, num_minutiae, num_samples, fused, source_count, 
                    consensus_score)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, name, protected_hash, bit_length, minutiae_json,
                 quality, num_minutiae, num_samples, fused, source_count,
                 consensus_score)
            )
            self.conn.commit()
            
            self._log_audit(
                username,
                "FINGERPRINT_ENROLLED",
                f"user_id={user_id}, name={name}, quality={quality:.3f}",
                "success"
            )
            return True
        
        except sqlite.IntegrityError:
            self._log_audit(username, "FINGERPRINT_ENROLL_FAILED", f"user_id={user_id} already exists", "failure")
            return False
    
    def get_fingerprint(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get fingerprint template.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with template info or None
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM fingerprints WHERE user_id = ?", (user_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        data = dict(row)
        
        # Deserialize template SECURELY (Protected + Minutiae only)
        # NO PICKLE FALLBACK - Security first!
        try:
            if data.get("protected_hash") and data.get("minutiae_json"):
                from src.models_serialization import template_from_secure_dict
                import json
                
                secure_dict = {
                    'identifier': data['user_id'],
                    'image_path': '',  # Not stored in DB
                    'protected': data['protected_hash'],
                    'bit_length': data['bit_length'],
                    'minutiae': json.loads(data['minutiae_json']),
                    'quality': data['quality'],
                    'fused': bool(data.get('fused', False)),
                    'source_count': data.get('source_count', 1),
                    'consensus_score': data.get('consensus_score', 1.0)
                }
                data["template"] = template_from_secure_dict(secure_dict)
            else:
                # No secure data available - template cannot be loaded
                print(f"[ERROR] User {data['user_id']} has no secure template data (legacy format?)")
                data["template"] = None
        except Exception as e:
            # If deserialization fails, set template to None but keep metadata
            print(f"[WARNING] Failed to deserialize template for {data['user_id']}: {e}")
            data["template"] = None
        
        return data
    
    def list_fingerprints(self) -> List[Dict[str, Any]]:
        """List all fingerprints (without template blobs).
        
        Returns:
            List of dicts with metadata
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT id, user_id, name, quality, num_minutiae, num_samples, 
                      created_at, updated_at 
               FROM fingerprints 
               ORDER BY created_at DESC"""
        )
        
        return [dict(row) for row in cursor.fetchall()]
    
    def update_fingerprint(
        self,
        user_id: str,
        name: str = None,
        template_obj: Any = None,
        username: str = "system"
    ) -> bool:
        """Update fingerprint.
        
        Args:
            user_id: User identifier
            name: New name (optional)
            template_obj: New template (optional)
            username: Who performed the operation
            
        Returns:
            True if updated successfully
        """
        cursor = self.conn.cursor()
        
        # Check if exists
        cursor.execute("SELECT id FROM fingerprints WHERE user_id = ?", (user_id,))
        if not cursor.fetchone():
            return False
        
        # Build update query
        updates = []
        params = []
        
        if name:
            updates.append("name = ?")
            params.append(name)
        
        if template_obj:
            # Serialize template SECURELY (Protected + Minutiae only)
            from src.models_serialization import template_to_secure_dict
            import json
            
            secure_data = template_to_secure_dict(template_obj)
            
            updates.append("protected_hash = ?")
            params.append(secure_data['protected'])
            
            updates.append("bit_length = ?")
            params.append(secure_data['bit_length'])
            
            updates.append("minutiae_json = ?")
            params.append(json.dumps(secure_data['minutiae']))
            
            updates.append("quality = ?")
            params.append(secure_data['quality'])
            
            updates.append("num_minutiae = ?")
            params.append(len(secure_data['minutiae']))
            
            updates.append("num_samples = ?")
            params.append(secure_data['source_count'])
            
            updates.append("fused = ?")
            params.append(secure_data['fused'])
            
            updates.append("source_count = ?")
            params.append(secure_data['source_count'])
            
            updates.append("consensus_score = ?")
            params.append(secure_data['consensus_score'])
        
        updates.append("updated_at = ?")
        params.append(datetime.now(timezone.utc).isoformat())
        
        params.append(user_id)
        
        cursor.execute(
            f"UPDATE fingerprints SET {', '.join(updates)} WHERE user_id = ?",
            params
        )
        self.conn.commit()
        
        self._log_audit(username, "FINGERPRINT_UPDATED", f"user_id={user_id}", "success")
        return True
    
    def delete_fingerprint(self, user_id: str, username: str = "system") -> bool:
        """Delete fingerprint.
        
        Args:
            user_id: User identifier
            username: Who performed the operation
            
        Returns:
            True if deleted successfully
        """
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM fingerprints WHERE user_id = ?", (user_id,))
        deleted = cursor.rowcount > 0
        self.conn.commit()
        
        if deleted:
            self._log_audit(username, "FINGERPRINT_DELETED", f"user_id={user_id}", "success")
        
        return deleted
    
    def delete_all_fingerprints(self, username: str = "system") -> int:
        """Delete all fingerprints.
        
        Args:
            username: Who performed the operation
            
        Returns:
            Number of deleted records
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM fingerprints")
        count = cursor.fetchone()[0]
        
        cursor.execute("DELETE FROM fingerprints")
        self.conn.commit()
        
        self._log_audit(username, "ALL_FINGERPRINTS_DELETED", f"count={count}", "success")
        return count
    
    def get_all_templates(self) -> List[Tuple[str, Any]]:
        """Get all templates for caching.
        
        Returns:
            List of (user_id, template_object) tuples
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT user_id, protected_hash, bit_length, minutiae_json, 
                   quality, fused, source_count, consensus_score
            FROM fingerprints
        """)
        
        from src.models_serialization import template_from_secure_dict
        import json
        
        templates = []
        for row in cursor.fetchall():
            try:
                # Load ONLY secure format (Protected + Minutiae)
                # NO PICKLE FALLBACK - Security first!
                if row["protected_hash"] and row["minutiae_json"]:
                    secure_dict = {
                        'identifier': row['user_id'],
                        'image_path': '',
                        'protected': row['protected_hash'],
                        'bit_length': row['bit_length'],
                        'minutiae': json.loads(row['minutiae_json']),
                        'quality': row['quality'],
                        'fused': bool(row.get('fused', False)),
                        'source_count': row.get('source_count', 1),
                        'consensus_score': row.get('consensus_score', 1.0)
                    }
                    template = template_from_secure_dict(secure_dict)
                else:
                    # Skip rows without secure template data
                    print(f"[WARNING] Skipping user {row['user_id']} - no secure template data (legacy format?)")
                    continue
                
                templates.append((row["user_id"], template))
            except Exception as e:
                # Skip problematic templates but log the error
                print(f"[WARNING] Failed to load template for {row['user_id']}: {e}")
                continue
        
        return templates
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics.
        
        Returns:
            Dict with statistics
        """
        cursor = self.conn.cursor()
        
        # Fingerprint stats
        cursor.execute("SELECT COUNT(*) as count FROM fingerprints")
        num_users = cursor.fetchone()["count"]
        
        cursor.execute("SELECT AVG(quality) as avg_quality FROM fingerprints")
        avg_quality = cursor.fetchone()["avg_quality"] or 0.0
        
        cursor.execute("SELECT SUM(num_minutiae) as total_minutiae FROM fingerprints")
        total_minutiae = cursor.fetchone()["total_minutiae"] or 0
        
        # Auth stats
        cursor.execute("SELECT COUNT(*) as count FROM auth_users")
        num_auth_users = cursor.fetchone()["count"]
        
        return {
            "num_users": num_users,
            "avg_template_quality": avg_quality,
            "total_minutiae": total_minutiae,
            "num_auth_users": num_auth_users,
            "encrypted": self.encrypted
        }
    
    # ========================================================================
    # AUDIT LOGGING
    # ========================================================================
    
    def _log_audit(
        self,
        username: str,
        action: str,
        details: str = "",
        result: str = "success",
        ip_address: str = None
    ):
        """Log audit entry.
        
        Args:
            username: Username performing action
            action: Action type
            details: Additional details
            result: 'success' or 'failure'
            ip_address: Client IP address
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO audit_log (username, action, details, result, ip_address)
               VALUES (?, ?, ?, ?, ?)""",
            (username, action, details, result, ip_address)
        )
        self.conn.commit()
    
    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries.
        
        Args:
            limit: Maximum number of entries
            
        Returns:
            List of audit entries
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        
        return [dict(row) for row in cursor.fetchall()]
    
    # ========================================================================
    # JOB MANAGEMENT
    # ========================================================================
    
    def create_job(self, job_id: str, job_type: str, created_by: str) -> bool:
        """Create background job entry.
        
        Args:
            job_id: Unique job identifier
            job_type: Type of job (e.g., 'load_folder')
            created_by: Username who created the job
            
        Returns:
            True if created successfully
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """INSERT INTO jobs (id, type, status, created_by, started_at)
                   VALUES (?, ?, 'pending', ?, ?)""",
                (job_id, job_type, created_by, datetime.now(timezone.utc).isoformat())
            )
            self.conn.commit()
            return True
        except sqlite.IntegrityError:
            return False
    
    def update_job(
        self,
        job_id: str,
        status: str = None,
        progress: str = None,
        result: str = None,
        error: str = None
    ) -> bool:
        """Update job status.
        
        Args:
            job_id: Job identifier
            status: New status ('pending', 'processing', 'completed', 'failed')
            progress: Progress info (JSON string)
            result: Result data (JSON string)
            error: Error message
            
        Returns:
            True if updated successfully
        """
        updates = []
        params = []
        
        if status:
            updates.append("status = ?")
            params.append(status)
            
            if status == 'completed' or status == 'failed':
                updates.append("completed_at = ?")
                params.append(datetime.now(timezone.utc).isoformat())
        
        if progress is not None:
            updates.append("progress = ?")
            params.append(progress)
        
        if result is not None:
            updates.append("result = ?")
            params.append(result)
        
        if error is not None:
            updates.append("error = ?")
            params.append(error)
        
        if not updates:
            return False
        
        params.append(job_id)
        
        cursor = self.conn.cursor()
        cursor.execute(
            f"UPDATE jobs SET {', '.join(updates)} WHERE id = ?",
            params
        )
        updated = cursor.rowcount > 0
        self.conn.commit()
        
        return updated
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Dict with job info or None
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def cleanup_old_jobs(self, hours: int = 24) -> int:
        """Delete old completed jobs.
        
        Args:
            hours: Delete jobs older than this many hours
            
        Returns:
            Number of deleted jobs
        """
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        cursor = self.conn.cursor()
        cursor.execute(
            """DELETE FROM jobs 
               WHERE status IN ('completed', 'failed') 
               AND completed_at < ?""",
            (cutoff.isoformat(),)
        )
        deleted = cursor.rowcount
        self.conn.commit()
        
        return deleted
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
