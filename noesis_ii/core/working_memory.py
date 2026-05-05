import sqlite3
import os
import re
import math
from datetime import datetime, timedelta, timezone

# ------------------------------------------------------------------ #
# Conflict Resolution Operation Types (aligned with Mem0's ADD/UPDATE/DELETE/NOOP)
# ------------------------------------------------------------------ #
MEMORY_OP_ADD    = "ADD"     # New content, normal write
MEMORY_OP_UPDATE = "UPDATE"  # Similar content, supplement/merge
MEMORY_OP_SKIP   = "SKIP"    # Highly duplicate, skip

# Similarity Thresholds
_SIM_SKIP_THRESHOLD   = 0.85   # > 0.85 considered duplicate, SKIP directly
_SIM_UPDATE_THRESHOLD = 0.45   # 0.45-0.85 considered similar, UPDATE append

def _tokenize(text: str):
    """Simple tokenization: Chinese split by character, English split by space"""
    return re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text.lower())

def _tf(tokens: list) -> dict:
    """Calculate term frequency (TF)"""
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    total = len(tokens) or 1
    return {t: c / total for t, c in tf.items()}

def _cosine_sim(vec1: dict, vec2: dict) -> float:
    """TF-weighted cosine similarity"""
    common = set(vec1) & set(vec2)
    if not common:
        return 0.0
    dot = sum(vec1[t] * vec2[t] for t in common)
    n1 = math.sqrt(sum(v * v for v in vec1.values()))
    n2 = math.sqrt(sum(v * v for v in vec2.values()))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


class WorkingMemory:
    """
    Working Memory Module
    Provides short-term fast memory storage for immediate content processing.
    Acts as a cache layer before consolidation to long-term memory.
    """

    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        """Connect to the SQLite database"""
        dir_name = os.path.dirname(self.db_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self.conn

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    # ------------------------------------------------------------------ #
    # Core Conflict Resolution
    # ------------------------------------------------------------------ #

    def _decide_operation(self, content: str) -> tuple:
        """
        Decide operation type: ADD / UPDATE / SKIP

        Returns:
            (op: str, match_id: int|None, sim: float)
        """
        conn = self.connect()
        try:
            cursor = conn.cursor()
            # Only compare unconsolidated memories from the last 48 hours
            cursor.execute('''
                SELECT id, content FROM working_memory
                WHERE is_consolidated = 0
                ORDER BY timestamp DESC
                LIMIT 50
            ''')
            recent = [dict(row) for row in cursor.fetchall()]
        finally:
            self.close()

        if not recent:
            return MEMORY_OP_ADD, None, 0.0

        new_tf = _tf(_tokenize(content))
        best_sim = 0.0
        best_id  = None

        for row in recent:
            existing_tf = _tf(_tokenize(str(row['content'])))
            sim = _cosine_sim(new_tf, existing_tf)
            if sim > best_sim:
                best_sim = sim
                best_id  = row['id']

        if best_sim >= _SIM_SKIP_THRESHOLD:
            return MEMORY_OP_SKIP, best_id, best_sim
        elif best_sim >= _SIM_UPDATE_THRESHOLD:
            return MEMORY_OP_UPDATE, best_id, best_sim
        else:
            return MEMORY_OP_ADD, None, best_sim

    def capture(self, content, emotion=None, conflict_check=True):
        """
        Capture experience to working memory (with conflict resolution)

        Args:
            content:        Experience content
            emotion:        Emotion label
            conflict_check: Enable conflict detection (default: enabled)

        Returns:
            (entry_id: int|None, op: str)
            - entry_id: Written record ID (SKIP returns the matched ID)
            - op: ADD / UPDATE / SKIP
        """
        if conflict_check:
            op, match_id, sim = self._decide_operation(content)
        else:
            op, match_id, sim = MEMORY_OP_ADD, None, 0.0

        if op == MEMORY_OP_SKIP:
            print(f"[WorkingMemory] SKIP: similarity {sim:.2f} > {_SIM_SKIP_THRESHOLD}, duplicate with id={match_id}")
            return match_id, op

        conn = self.connect()
        try:
            cursor = conn.cursor()

            if op == MEMORY_OP_UPDATE and match_id is not None:
                # Append new content to existing entry (separated by \n---\n)
                cursor.execute('SELECT content FROM working_memory WHERE id = ?', (match_id,))
                row = cursor.fetchone()
                if row:
                    merged = str(row['content']) + '\n---\n' + content
                    cursor.execute(
                        'UPDATE working_memory SET content = ?, timestamp = CURRENT_TIMESTAMP WHERE id = ?',
                        (merged, match_id)
                    )
                    conn.commit()
                    print(f"[WorkingMemory] UPDATE: similarity {sim:.2f}, content appended to id={match_id}")
                    return match_id, op
                # Fallback to ADD if fetch fails
                op = MEMORY_OP_ADD

            # ADD: Normal insert
            cursor.execute('''
            INSERT INTO working_memory (content, emotion, timestamp, is_consolidated, ttl)
            VALUES (?, ?, CURRENT_TIMESTAMP, 0, 172800)
            ''', (content, emotion))

            conn.commit()
            entry_id = cursor.lastrowid
            if op == MEMORY_OP_ADD:
                print(f"[WorkingMemory] ADD: id={entry_id} (sim_max={sim:.2f})")
            return entry_id, op
        finally:
            self.close()

    # ------------------------------------------------------------------ #
    # L1 Hot Cache Layer (Recent N conversation rounds, directly concatenated to context)
    # ------------------------------------------------------------------ #

    def get_hot_context(self, n: int = 5, format: str = "text") -> str:
        """
        Get L1 hot cache: Most recent n working memory entries (regardless of consolidation status).

        Design rationale:
          - WorkingMemory full data = L1 (hot cache) + pending queue
          - L1 hot cache directly concatenated to generation context, no vector retrieval
          - Reduces information loss of "recently said X but LTM hasn't retrieved yet"

        Args:
            n:      Return most recent n records (default: 5)
            format: "text" returns concatenated string, "list" returns dict list

        Returns:
            format="text": "[L1-Hot] content1\n[L1-Hot] content2\n..."
            format="list":  [{"id": ..., "content": ..., "timestamp": ...}, ...]
        """
        conn = self.connect()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, content, emotion, timestamp, is_consolidated
                FROM working_memory
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (n,))
            rows = [dict(row) for row in cursor.fetchall()]
        finally:
            self.close()

        if not rows:
            return "" if format == "text" else []

        # Reverse to chronological order (newest at end)
        rows = list(reversed(rows))

        if format == "list":
            return rows

        parts = []
        for r in rows:
            ts = r.get('timestamp', '')
            ts_short = str(ts)[:16] if ts else ''
            parts.append(f"[L1 {ts_short}] {r['content']}")
        return '\n'.join(parts)

    def get_pending(self, limit=10):
        """Get pending working memory entries to be consolidated"""
        conn = self.connect()
        try:
            cursor = conn.cursor()

            cursor.execute('''
            SELECT * FROM working_memory
            WHERE is_consolidated = 0
            ORDER BY timestamp ASC
            LIMIT ?
            ''', (limit,))

            entries = [dict(row) for row in cursor.fetchall()]
            return entries
        finally:
            self.close()

    def mark_consolidated(self, entry_id):
        """Mark entry as consolidated"""
        conn = self.connect()
        try:
            cursor = conn.cursor()

            cursor.execute('''
            UPDATE working_memory
            SET is_consolidated = 1
            WHERE id = ?
            ''', (entry_id,))

            conn.commit()
            affected = cursor.rowcount
            return affected > 0
        finally:
            self.close()

    def expire_old_entries(self):
        """
        Soft delete expired entries: active → dormant → forgotten

        Three-stage forgetting (inspired by Kimi Claw's memory decay model):
        1. active → dormant: Enter dormant state after TTL expires, excluded from normal retrieval but data retained
        2. dormant → forgotten: Enter forgotten state after 7 days of dormancy
        3. forgotten: Hard delete available after 30 days (not auto-executed, requires manual cleanup)
        """
        import time as _time
        now = _time.time()
        conn = self.connect()
        try:
            cursor = conn.cursor()

            # Stage 1: active → dormant (TTL expired)
            cursor.execute('''
                UPDATE working_memory
                SET memory_state = 'dormant',
                    dormant_since = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE memory_state = 'active'
                  AND is_consolidated = 1
                  AND (julianday('now') - julianday(timestamp)) * 86400 > ttl
            ''', (now,))
            active_to_dormant = cursor.rowcount

            # Stage 2: dormant → forgotten (7 days of dormancy)
            cursor.execute('''
                UPDATE working_memory
                SET memory_state = 'forgotten',
                    updated_at = CURRENT_TIMESTAMP
                WHERE memory_state = 'dormant'
                  AND (julianday('now') - julianday(timestamp)) * 86400 > (ttl + 604800)
            ''')
            dormant_to_forgotten = cursor.rowcount

            conn.commit()
            total = active_to_dormant + dormant_to_forgotten
            if total > 0:
                print(f"[WorkingMemory] Soft-forget: {active_to_dormant} active→dormant, "
                      f"{dormant_to_forgotten} dormant→forgotten")
            return total
        finally:
            self.close()

    def get_active_entries(self, limit=100):
        """Get only active state working memory entries (exclude dormant/forgotten)"""
        conn = self.connect()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM working_memory
                WHERE memory_state = 'active'
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            self.close()

    def recover_dormant(self, entry_id):
        """Recover from dormant/forgotten state to active (simulating memory recall)"""
        conn = self.connect()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE working_memory
                SET memory_state = 'active',
                    dormant_since = 0,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND memory_state IN ('dormant', 'forgotten')
            ''', (entry_id,))
            conn.commit()
            affected = cursor.rowcount
            if affected > 0:
                print(f"[WorkingMemory] Recovered entry id={entry_id} to active")
            return affected > 0
        finally:
            self.close()

    def purge_forgotten(self, days=30):
        """Hard delete forgotten entries older than specified days (manual call required, irreversible)"""
        conn = self.connect()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM working_memory
                WHERE memory_state = 'forgotten'
                  AND (julianday('now') - julianday(timestamp)) * 86400 > ?
            ''', (days * 86400,))
            conn.commit()
            deleted = cursor.rowcount
            if deleted > 0:
                print(f"[WorkingMemory] Purged {deleted} forgotten entries older than {days} days")
            return deleted
        finally:
            self.close()

    def get_all(self, limit=100):
        """Get all working memory entries"""
        conn = self.connect()
        try:
            cursor = conn.cursor()

            cursor.execute('''
            SELECT * FROM working_memory
            ORDER BY timestamp DESC
            LIMIT ?
            ''', (limit,))

            entries = [dict(row) for row in cursor.fetchall()]
            return entries
        finally:
            self.close()

    def get_by_id(self, entry_id):
        """Get working memory entry by ID"""
        conn = self.connect()
        try:
            cursor = conn.cursor()

            cursor.execute('''
            SELECT * FROM working_memory
            WHERE id = ?
            ''', (entry_id,))

            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            self.close()

    def delete(self, entry_id):
        """Delete working memory entry"""
        conn = self.connect()
        try:
            cursor = conn.cursor()

            cursor.execute('''
            DELETE FROM working_memory
            WHERE id = ?
            ''', (entry_id,))

            conn.commit()
            affected = cursor.rowcount
            return affected > 0
        finally:
            self.close()
