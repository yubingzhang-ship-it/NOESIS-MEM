import sqlite3
import os
import re
import math
from datetime import datetime, timedelta, timezone

# ------------------------------------------------------------------ #
# 冲突消解操作类型（对标 Mem0 的 ADD/UPDATE/DELETE/NOOP）
# ------------------------------------------------------------------ #
MEMORY_OP_ADD    = "ADD"     # 新内容，正常写入
MEMORY_OP_UPDATE = "UPDATE"  # 近似内容，补充/合并
MEMORY_OP_SKIP   = "SKIP"    # 高度重复，跳过

# 相似度阈值
_SIM_SKIP_THRESHOLD   = 0.85   # > 0.85 视为重复，直接 SKIP
_SIM_UPDATE_THRESHOLD = 0.45   # 0.45-0.85 视为近似，UPDATE 追加

def _tokenize(text: str):
    """简单分词：中文按字拆分，英文按空格"""
    return re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text.lower())

def _tf(tokens: list) -> dict:
    """计算词频 TF"""
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    total = len(tokens) or 1
    return {t: c / total for t, c in tf.items()}

def _cosine_sim(vec1: dict, vec2: dict) -> float:
    """TF 加权余弦相似度"""
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

    # ------------------------------------------------------------------ #
    # 冲突消解核心
    # ------------------------------------------------------------------ #

    def _decide_operation(self, content: str) -> tuple:
        """
        决策操作类型：ADD / UPDATE / SKIP
        
        Returns:
            (op: str, match_id: int|None, sim: float)
        """
        conn = self.connect()
        try:
            cursor = conn.cursor()
            # 只比对最近 48 小时未整理的记忆（避免对 LTM 产生影响）
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
        捕获经验到工作记忆（含冲突消解）
        
        Args:
            content:        经验内容
            emotion:        情绪标签
            conflict_check: 是否启用冲突检测（默认开启）
        
        Returns:
            (entry_id: int|None, op: str)
            - entry_id: 写入的记录 ID（SKIP 时为被命中的 ID）
            - op: ADD / UPDATE / SKIP
        """
        if conflict_check:
            op, match_id, sim = self._decide_operation(content)
        else:
            op, match_id, sim = MEMORY_OP_ADD, None, 0.0

        if op == MEMORY_OP_SKIP:
            print(f"[WorkingMemory] SKIP: 相似度 {sim:.2f} > {_SIM_SKIP_THRESHOLD}，与 id={match_id} 内容重复")
            return match_id, op

        conn = self.connect()
        try:
            cursor = conn.cursor()

            if op == MEMORY_OP_UPDATE and match_id is not None:
                # 将新内容追加到已有条目（用 \n---\n 分隔）
                cursor.execute('SELECT content FROM working_memory WHERE id = ?', (match_id,))
                row = cursor.fetchone()
                if row:
                    merged = str(row['content']) + '\n---\n' + content
                    cursor.execute(
                        'UPDATE working_memory SET content = ?, timestamp = CURRENT_TIMESTAMP WHERE id = ?',
                        (merged, match_id)
                    )
                    conn.commit()
                    print(f"[WorkingMemory] UPDATE: 相似度 {sim:.2f}，内容追加到 id={match_id}")
                    return match_id, op
                # 如果 fetch 失败则降级为 ADD
                op = MEMORY_OP_ADD

            # ADD：正常插入
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
    # L1 热缓存层（最近 N 轮对话，不参与 Embedding，直接拼接进上下文）
    # ------------------------------------------------------------------ #

    def get_hot_context(self, n: int = 5, format: str = "text") -> str:
        """
        获取 L1 热缓存：最近 n 条工作记忆（无论是否已整理）。
        
        设计意图：
          - WorkingMemory 全量数据 = L1（热缓存）+ 待整理队列
          - L1 热缓存直接拼接进生成上下文，不走向量检索
          - 减少"最近说过 X 但 LTM 还没检索到"的信息丢失
        
        Args:
            n:      返回最近 n 条记录（默认 5）
            format: "text" 返回拼接字符串，"list" 返回 dict 列表
        
        Returns:
            format="text": "[L1-Hot] 内容1\n[L1-Hot] 内容2\n..."
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

        # 按时间升序（最新的在后）
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
        """获取待整理的工作记忆条目"""
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
        """标记为已整理"""
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
        """软删除过期条目：active → dormant → forgotten

        遗忘三阶段（借鉴 Kimi Claw 的记忆衰减模型）：
        1. active → dormant: TTL 过期后进入休眠，不参与常规检索但保留数据
        2. dormant → forgotten: 休眠 7 天后进入遗忘状态
        3. forgotten: 30 天后可硬删除（当前不自动执行，需手动清理）
        """
        import time as _time
        now = _time.time()
        conn = self.connect()
        try:
            cursor = conn.cursor()

            # 阶段1：active → dormant（TTL 过期）
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

            # 阶段2：dormant → forgotten（休眠 7 天）
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
        """只获取 active 状态的工作记忆（排除 dormant/forgotten）"""
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
        """从 dormant/forgotten 状态恢复为 active（模拟记忆唤醒）"""
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
        """硬删除 forgotten 超过指定天数的条目（需手动调用，不可逆）"""
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
        """获取所有工作记忆条目"""
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
        """根据ID获取工作记忆条目"""
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
        """删除工作记忆条目"""
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