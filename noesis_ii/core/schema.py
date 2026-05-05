import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any

class Schema:
    """Database schema initialization and management for NOESIS-II."""
    
    def __init__(self, db_path: str = 'data/noesis.db'):
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
    
    def create_tables(self):
        """Create all necessary database tables."""
        self.create_working_memory_table()
        self.create_ltm_nodes_table()
        self.create_ltm_links_table()
        self.create_memory_traces_table()
        self.connection.commit()
    
    def create_working_memory_table(self):
        """Create working_memory table for short-term fast memory."""
        query = '''
        CREATE TABLE IF NOT EXISTS working_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            emotion TEXT,
            is_consolidated BOOLEAN DEFAULT 0,
            ttl INTEGER,
            memory_state TEXT DEFAULT 'active',
            dormant_since DATETIME,
            emotion_data TEXT
        )
        '''
        self.cursor.execute(query)
    
    def create_ltm_nodes_table(self):
        """Create ltm_nodes table for long-term memory network storage."""
        query = '''
        CREATE TABLE IF NOT EXISTS ltm_nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            type TEXT NOT NULL,
            weight REAL DEFAULT 1.0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_accessed DATETIME,
            raw_anchors TEXT
        )
        '''
        self.cursor.execute(query)
    
    def create_ltm_links_table(self):
        """Create ltm_links table for memory node associations."""
        query = '''
        CREATE TABLE IF NOT EXISTS ltm_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_node_id INTEGER NOT NULL,
            target_node_id INTEGER NOT NULL,
            strength REAL DEFAULT 1.0,
            relation_type TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_node_id) REFERENCES ltm_nodes(id),
            FOREIGN KEY (target_node_id) REFERENCES ltm_nodes(id)
        )
        '''
        self.cursor.execute(query)
    
    def create_memory_traces_table(self):
        """Create memory_traces table for personality consistency management."""
        query = '''
        CREATE TABLE IF NOT EXISTS memory_traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT UNIQUE,
            content_summary TEXT NOT NULL,
            trace_type TEXT,
            strength REAL DEFAULT 1.0,
            long_term_impact TEXT,
            condition_pattern TEXT,
            access_history TEXT,
            self_relevance REAL,
            last_accessed DATETIME,
            access_count INTEGER DEFAULT 0,
            is_active BOOLEAN DEFAULT 1,
            memory_state TEXT DEFAULT 'active',
            dormant_since DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            emotion_data TEXT
        )
        '''
        self.cursor.execute(query)
    
    def close(self):
        """Close database connection."""
        self.connection.close()