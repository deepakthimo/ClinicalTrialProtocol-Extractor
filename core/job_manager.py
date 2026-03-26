import json
import sqlite3
import os
from enum import Enum
from typing import Dict, Optional, Any, List
from pydantic import BaseModel

class JobStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class JobRecord(BaseModel):
    job_id: str
    status: JobStatus
    pdf_url: str
    sponsor_name: str
    result_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str

class JobManager:
    """
    A SQLite-backed in-memory job manager to track the state of API-submitted PDF extractions.
    Ensures state is resilient across server restarts.
    """
    def __init__(self, db_path: str = r"C:\Users\DeepakTM\Music\Projects\lilly-pdf-extractor-agent\memory\job_states.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    pdf_url TEXT NOT NULL,
                    sponsor_name TEXT NOT NULL,
                    result_path TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def _row_to_record(self, row) -> JobRecord:
        # Gracefully handle status loading
        try:
            status_enum = JobStatus(row[1])
        except ValueError:
            status_enum = JobStatus.FAILED

        return JobRecord(
            job_id=row[0],
            status=status_enum,
            pdf_url=row[2],
            sponsor_name=row[3],
            result_path=row[4],
            error_message=row[5],
            created_at=row[6]
        )

    def create_job(self, record: JobRecord) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO jobs (job_id, status, pdf_url, sponsor_name, result_path, error_message, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                record.job_id, record.status.value, record.pdf_url, record.sponsor_name,
                record.result_path, record.error_message, record.created_at
            ))
            conn.commit()

    def update_job_status(self, job_id: str, status: JobStatus, result_path: str = None, error: str = None) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            updates = ["status = ?"]
            params = [status.value]
            
            if result_path is not None:
                updates.append("result_path = ?")
                params.append(result_path)
            if error is not None:
                updates.append("error_message = ?")
                params.append(error)
                
            params.append(job_id)
            
            query = f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?"
            cursor.execute(query, tuple(params))
            conn.commit()

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_record(row)
        return None

    def get_incomplete_jobs(self) -> List[JobRecord]:
        """Returns jobs that are stuck in PENDING or IN_PROGRESS state."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM jobs WHERE status IN (?, ?, ?)", 
                           (JobStatus.PENDING.value, JobStatus.IN_PROGRESS.value, JobStatus.FAILED.value)) # Added FAILED to retry failed jobs
            rows = cursor.fetchall()
            return [self._row_to_record(row) for row in rows]

# Singleton instance for the application
job_manager = JobManager()