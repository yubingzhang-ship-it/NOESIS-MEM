import sqlite3
import os


class Schema:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        """连接到数据库"""
        dir_name = os.path.dirname(self.db_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self.conn

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()

    def create_tables(self):
        """创建所有数据表"""
        conn = self.connect()
        cursor = conn.cursor()

        # 工作记忆表
        # memory_state: active → dormant → forgotten（软删除替代硬删除）
        # emotion_data: 结构化情绪向量 JSON {valence, arousal, dominant, tags}
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

        # 长期记忆节点表
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

        # 长期记忆关联表
        # relation_type: 关系类型（causal/associated/similar/contrast/temporal/related）
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

        # 人格画像表（替代原 deep_personality）
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

        # 记忆痕迹表（替代原 alaya_seeds，路线A重构）
        # memory_state: active(正常) → dormant(休眠，低权重不检索但保留) → forgotten(遗忘，可恢复)
        # emotion_data: 结构化情绪向量 JSON {valence, arousal, dominant, tags, narrative_hook}
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

        # 活动事件日志表（替代原 conscious_events）
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS activity_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            relevance_score REAL
        )
        ''')

        # HGM 状态表
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

        # 动作队列表（延迟动作调度）
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

        # P1 新增：延展心智外部资源表
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

        # 记忆痕迹关联表（替代原 seed_links）
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trace_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_trace_id INTEGER NOT NULL REFERENCES memory_traces(id),
            target_trace_id INTEGER NOT NULL REFERENCES memory_traces(id),
            relation_type TEXT NOT NULL DEFAULT \'related\',
            strength REAL NOT NULL DEFAULT 0.5,
            context TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_trace_id, target_trace_id, relation_type)
        )
        ''')

        conn.commit()
        self.close()

    def _get_table_columns(self, conn, table_name):
        """获取表的列名列表"""
        # 防止SQL注入：只允许字母、数字、下划线
        import re
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name):
            raise ValueError(f"Invalid table name: {table_name}")
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        return [row[1] for row in cursor]

    def _migrate_alaya_seeds(self, conn):
        """迁移旧 alaya_seeds 表到新 memory_traces 结构（路线A）"""
        # 检查旧表是否存在
        old_columns = []
        try:
            old_columns = self._get_table_columns(conn, 'alaya_seeds')
        except Exception:
            pass  # 旧表不存在，无需迁移

        if not old_columns:
            return  # 没有旧数据

        # 检查新表是否已有数据
        new_columns = self._get_table_columns(conn, 'memory_traces')

        # 备份旧数据
        try:
            # 过滤非法列名
            import re
            safe_columns = [c for c in old_columns if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', c)]
            if not safe_columns:
                return
            col_list = ', '.join(safe_columns)
            conn.execute(f'CREATE TABLE IF NOT EXISTS _legacy_seeds_backup AS SELECT {col_list} FROM alaya_seeds')
        except Exception:
            pass

        # 迁移到新表结构
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

        # 清理（不删除旧表，以防回滚）
        try:
            conn.execute('DROP TABLE IF EXISTS _legacy_seeds_backup')
        except Exception:
            pass

    def _migrate_deep_personality(self, conn):
        """迁移旧 deep_personality 表到新 persona_profiles 结构（路线A）"""
        old_columns = []
        try:
            old_columns = self._get_table_columns(conn, 'deep_personality')
        except Exception:
            pass

        if not old_columns:
            return

        try:
            # 过滤非法列名
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
        """迁移旧 vipaka_queue 表名到 action_queue（路线A术语清理）"""
        try:
            old_columns = self._get_table_columns(conn, 'vipaka_queue')
            if old_columns:
                conn.execute('ALTER TABLE vipaka_queue RENAME TO action_queue')
                print("[MIGRATE] vipaka_queue -> action_queue: table renamed")
        except Exception:
            pass  # 不存在或已迁移

    def init_db(self):
        """初始化数据库（含自动迁移）"""
        conn = self.connect()
        self._migrate_alaya_seeds(conn)
        self._migrate_deep_personality(conn)
        self._migrate_vipaka_queue(conn)
        self.close()
        self.create_tables()

    def migrate(self):
        """数据库迁移"""
        self.init_db()
        self._migrate_soft_forget()
        self._migrate_relation_type()

    def _migrate_relation_type(self):
        """为 ltm_links 添加 relation_type 字段（改进2：图关联增强）"""
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
        """为现有表添加软删除遗忘字段（memory_state, dormant_since）"""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            # memory_traces: 添加 memory_state 和 dormant_since
            cols = self._get_table_columns(conn, 'memory_traces')
            if 'memory_state' not in cols:
                cursor.execute("ALTER TABLE memory_traces ADD COLUMN memory_state TEXT DEFAULT 'active'")
                print("[MIGRATE] memory_traces: added memory_state column")
            if 'dormant_since' not in cols:
                cursor.execute("ALTER TABLE memory_traces ADD COLUMN dormant_since REAL DEFAULT 0.0")
                print("[MIGRATE] memory_traces: added dormant_since column")

            # working_memory: 添加 memory_state 和 dormant_since
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

            # memory_traces: 添加 emotion_data
            cols = self._get_table_columns(conn, 'memory_traces')
            if 'emotion_data' not in cols:
                cursor.execute("ALTER TABLE memory_traces ADD COLUMN emotion_data TEXT DEFAULT '{}'")
                print("[MIGRATE] memory_traces: added emotion_data column")

            conn.commit()
        except Exception as e:
            print(f"[MIGRATE] soft_forget migration warning: {e}")
        finally:
            self.close()
