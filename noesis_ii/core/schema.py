"""
Database Schema Module

Handles database initialization, table creation, and migration for NOESIS-II memory system.

Database Tables:
- working_memory: Short-term memory storage with TTL and soft-delete
- ltm_nodes: Long-term memory nodes with semantic content
- ltm_links: Associations between long-term memory nodes
- persona_profiles: Personality profiles (OCEAN model)
- memory_traces: Memory traces (replaces alaya_seeds)
- activity_events: Activity logging
- hgm_state: Hierarchical Generative Model state
- action_queue: Delayed action scheduling
- extended_resources: Extended mind external resources
- trace_links: Associations between memory traces

Migration History:
- Route A: Replace alaya_seeds with memory_traces, remove Buddhist terminology
- Add soft-delete fields (memory_state, dormant_since)
- Add emotion_data fields for narrative reconstruction
"""

import sqlite3
import os


class Schema:
    """
    Database Schema Manager
    Handles database initialization, table creation, and data migration.
    """

    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        """Connect to SQLite database"""
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

    def create_tables(self):
        """Create all database tables"""
        conn = self.connect()
        cursor = conn.cursor()

        # Working Memory Table
        # memory_state: active → dormant → forgotten (soft delete instead of hard delete)
        # emotion_data: Structured emotion vector JSON {valence, arousal, dominant, tags}
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS working_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            emotion TEXT,
            is_consolidated BOOLEAN DEFAULT 0,
            ttl INTEGER DEFAULT 172800,
            memory_state TEXT DEFAULT 'active',
            dormant_since REAL DEFAULT 0.0,
            emotion_data TEXT DEFAULT '{}'
        )
        ''')

        # Long-term Memory Nodes Table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS ltm_nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            type TEXT,
            weight REAL DEFAULT 1.0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Long-term Memory Links Table
        # relation_type: Relationship type (causal/associated/similar/contrast/temporal/related)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS ltm_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_node_id INTEGER,
            target_node_id INTEGER,
            strength REAL DEFAULT 0.5,
            relation_type TEXT DEFAULT 'related',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_node_id) REFERENCES ltm_nodes(id),
            FOREIGN KEY (target_node_id) REFERENCES ltm_nodes(id)
        )
        ''')

        # Persona Profiles Table (replaces deep_personality)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS persona_profiles (
            id TEXT PRIMARY KEY,
            dimension TEXT NOT NULL,
            content TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.5,
            source_node_ids TEXT,
            generated_at TEXT NOT NULL,
            valid_until TEXT,
            value REAL,
            description TEXT,
            last_updated TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Memory Traces Table (replaces alaya_seeds, Route A refactor)
        # memory_state: active(normal) → dormant(inactive but retained) → forgotten(can be recovered)
        # emotion_data: Structured emotion vector JSON {valence, arousal, dominant, tags, narrative_hook}
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS memory_traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT UNIQUE,
            content_summary TEXT,
            trace_type TEXT DEFAULT 'episodic',
            strength REAL DEFAULT 0.1,
            long_term_impact TEXT,
            condition_pattern TEXT,
            access_history TEXT DEFAULT '[]',
            self_relevance REAL DEFAULT 0.0,
            last_accessed REAL DEFAULT 0.0,
            access_count INTEGER DEFAULT 0,
            is_active INTEGER DEFAULT 1,
            memory_state TEXT DEFAULT 'active',
            dormant_since REAL DEFAULT 0.0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            emotion_data TEXT DEFAULT '{}'
        )
        ''')

        # Activity Events Log Table (replaces conscious_events)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS activity_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            relevance_score REAL
        )
        ''')

        # HGM State Table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS hgm_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            state_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS hgm_update_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            free_energy REAL,
            total_surprise REAL,
            learning_rate REAL,
            parameter_change REAL,
            emotion_intensity REAL,
            timestamp TEXT NOT NULL
        )
        ''')

        # Action Queue Table (delayed action scheduling)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS action_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            seed_db_id INTEGER,
            maturity_conditions TEXT,
            scheduled_at REAL,
            matured INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # P1 New: Extended Mind External Resources Table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS extended_resources (
            resource_id TEXT PRIMARY KEY,
            resource_type TEXT NOT NULL DEFAULT 'unknown',
            name TEXT DEFAULT '',
            description TEXT DEFAULT '',
            constant_access INTEGER DEFAULT 0,
            automatic_endorsement INTEGER DEFAULT 0,
            functional_integration REAL DEFAULT 0.0,
            trust_level REAL DEFAULT 0.5,
            coupling_history TEXT DEFAULT '[]',
            added_at TEXT,
            last_accessed TEXT
        )
        ''')

        # Memory Trace Links Table (replaces seed_links)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trace_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_trace_id INTEGER NOT NULL REFERENCES memory_traces(id),
            target_trace_id INTEGER NOT NULL REFERENCES memory_traces(id),
            relation_type TEXT NOT NULL DEFAULT 'related',
            strength REAL NOT NULL DEFAULT 0.5,
            context TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_trace_id, target_trace_id, relation_type)
        )
        ''')

        conn.commit()
        self.close()

    def _get_table_columns(self, conn, table_name):
        """Get column names for a table"""
        # Prevent SQL injection: only allow letters, numbers, underscore
        import re
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name):
            raise ValueError(f"Invalid table name: {table_name}")
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        return [row[1] for row in cursor]

    def _migrate_alaya_seeds(self, conn):
        """Migrate old alaya_seeds table to new memory_traces structure (Route A)"""
        old_columns = []
        try:
            old_columns = self._get_table_columns(conn, 'alaya_seeds')
        except Exception:
            pass  # Old table doesn't exist, no migration needed

        if not old_columns:
            return  # No legacy data

        new_columns = self._get_table_columns(conn, 'memory_traces')

        # Backup old data
        try:
            import re
            safe_columns = [c for c in old_columns if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', c)]
            if not safe_columns:
                return
            col_list = ', '.join(safe_columns)
            conn.execute(f'CREATE TABLE IF NOT EXISTS _legacy_seeds_backup AS SELECT {col_list} FROM alaya_seeds')
        except Exception:
            pass

        # Migrate to new structure
        try:
            conn.execute('''
            INSERT OR IGNORE INTO memory_traces (
                trace_id, content_summary, trace_type, strength,
                condition_pattern, self_relevance, last_accessed,
                access_count, is_active, created_at, updated_at
            )
            SELECT
                seed_id,
                COALESCE(content_summary, SUBSTR(content, 1, 200)),
                COALESCE(seed_type, 'episodic'),
                COALESCE(potency, 0.1),
                condition_pattern,
                COALESCE(self_relevance, 0.0),
                COALESCE(last_manifested, 0.0),
                COALESCE(manifestation_count, 0),
                CASE WHEN is_dormant = 1 THEN 0 ELSE 1 END,
                COALESCE(created_at, CURRENT_TIMESTAMP),
                COALESCE(updated_at, CURRENT_TIMESTAMP)
            FROM _legacy_seeds_backup
            ''')
            print("[MIGRATE] alaya_seeds -> memory_traces: legacy data migrated")
        except Exception as e:
            print(f"[MIGRATE] Warning: {e}")

        # Cleanup (don't delete old table in case of rollback)
        try:
            conn.execute('DROP TABLE IF EXISTS _legacy_seeds_backup')
        except Exception:
            pass

    def _migrate_deep_personality(self, conn):
        """Migrate old deep_personality table to new persona_profiles structure (Route A)"""
        old_columns = []
        try:
            old_columns = self._get_table_columns(conn, 'deep_personality')
        except Exception:
            pass

        if not old_columns:
            return

        try:
            import re
            safe_columns = [c for c in old_columns if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', c)]
            if not safe_columns:
                return
            col_list = ', '.join(safe_columns)
            conn.execute(f'CREATE TABLE IF NOT EXISTS _legacy_dp_backup AS SELECT {col_list} FROM deep_personality')
        except Exception:
            pass

        try:
            conn.execute('''
            INSERT OR IGNORE INTO persona_profiles (
                id, dimension, content, confidence, source_node_ids,
                generated_at, valid_until, value, description, last_updated
            )
            SELECT
                COALESCE(id, 'legacy_' || dimension || '_' || CAST(rowid AS TEXT)),
                dimension,
                content,
                COALESCE(confidence, 0.5),
                source_node_ids,
                COALESCE(generated_at, CURRENT_TIMESTAMP),
                COALESCE(valid_until, datetime('now', '+7 days')),
                value,
                description,
                COALESCE(last_updated, CURRENT_TIMESTAMP)
            FROM _legacy_dp_backup
            ''')
            print("[MIGRATE] deep_personality -> persona_profiles: legacy data migrated")
        except Exception as e:
            print(f"[MIGRATE] Warning: {e}")

        try:
            conn.execute('DROP TABLE IF EXISTS _legacy_dp_backup')
        except Exception:
            pass

    def _migrate_vipaka_queue(self, conn):
        """Rename old vipaka_queue table to action_queue (Route A terminology cleanup)"""
        try:
            old_columns = self._get_table_columns(conn, 'vipaka_queue')
            if old_columns:
                conn.execute('ALTER TABLE vipaka_queue RENAME TO action_queue')
                print("[MIGRATE] vipaka_queue -> action_queue: table renamed")
        except Exception:
            pass  # Doesn't exist or already migrated

    def init_db(self):
        """Initialize database (includes automatic migration)"""
        conn = self.connect()
        self._migrate_alaya_seeds(conn)
        self._migrate_deep_personality(conn)
        self._migrate_vipaka_queue(conn)
        self.close()
        self.create_tables()

    def migrate(self):
        """Execute database migrations"""
        self.init_db()
        self._migrate_soft_forget()
        self._migrate_relation_type()

    def _migrate_relation_type(self):
        """Add relation_type column to ltm_links (Improvement 2: Graph Association Enhancement)"""
        conn = self.connect()
        try:
            cols = self._get_table_columns(conn, 'ltm_links')
            if 'relation_type' not in cols:
                conn.execute("ALTER TABLE ltm_links ADD COLUMN relation_type TEXT DEFAULT 'related'")
                print("[MIGRATE] ltm_links: added relation_type column")
            conn.commit()
        except Exception as e:
            print(f"[MIGRATE] relation_type migration warning: {e}")
        finally:
            self.close()

    def _migrate_soft_forget(self):
        """Add soft-delete forget fields (memory_state, dormant_since) to existing tables"""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            # memory_traces: Add memory_state and dormant_since
            cols = self._get_table_columns(conn, 'memory_traces')
            if 'memory_state' not in cols:
                cursor.execute("ALTER TABLE memory_traces ADD COLUMN memory_state TEXT DEFAULT 'active'")
                print("[MIGRATE] memory_traces: added memory_state column")
            if 'dormant_since' not in cols:
                cursor.execute("ALTER TABLE memory_traces ADD COLUMN dormant_since REAL DEFAULT 0.0")
                print("[MIGRATE] memory_traces: added dormant_since column")

            # working_memory: Add memory_state and dormant_since
            cols = self._get_table_columns(conn, 'working_memory')
            if 'memory_state' not in cols:
                cursor.execute("ALTER TABLE working_memory ADD COLUMN memory_state TEXT DEFAULT 'active'")
                print("[MIGRATE] working_memory: added memory_state column")
            if 'dormant_since' not in cols:
                cursor.execute("ALTER TABLE working_memory ADD COLUMN dormant_since REAL DEFAULT 0.0")
                print("[MIGRATE] working_memory: added dormant_since column")
            if 'emotion_data' not in cols:
                cursor.execute("ALTER TABLE working_memory ADD COLUMN emotion_data TEXT DEFAULT '{}'")
                print("[MIGRATE] working_memory: added emotion_data column")

            # memory_traces: Add emotion_data
            cols = self._get_table_columns(conn, 'memory_traces')
            if 'emotion_data' not in cols:
                cursor.execute("ALTER TABLE memory_traces ADD COLUMN emotion_data TEXT DEFAULT '{}'")
                print("[MIGRATE] memory_traces: added emotion_data column")

            conn.commit()
        except Exception as e:
            print(f"[MIGRATE] soft_forget migration warning: {e}")
        finally:
            self.close()
