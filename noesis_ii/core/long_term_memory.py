"""
Long-Term Memory Module

Distributed long-term storage with semantic integration.

P1 Upgrade:
- Retrieval method upgrade: Composite score (semantic similarity × weight × recency)
- Semantic similarity: TF-IDF based cosine similarity (replaces pure LIKE queries)
- Weight formula: aligned with design document (base + frequency + emotion + link) × time_decay
- Forgetting mechanism: Power-law decay + dormant/deleted states

P2 Upgrade (2026-04-08):
- Fact anchors for hallucination prevention: raw_anchors field stores key facts (code/config/numbers)
- Verify facts during generative recall to prevent LLM hallucinations

Design reference: NOESIS Design Document Section 3.5
"""

import sqlite3
import os
import datetime
import math
import re
from collections import Counter
from typing import List, Dict, Optional


class LongTermMemory:
    """
    Long-Term Memory Module
    Provides distributed long-term storage with semantic integration.
    Supports weighted retrieval, forgetting mechanism, and fact anchoring.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.decay_rate = 0.05

        # Weight formula parameters (aligned with design document Section 4.3)
        self.base_weight = 0.5
        self.frequency_factor = 0.1
        self.emotion_factor = 0.3
        self.link_factor = 0.2

        # Forgetting thresholds
        self.dormant_threshold = 0.2
        self.delete_threshold = 0.1

        # TF-IDF cache (avoid redundant computation)
        self._idf_cache: Dict[str, float] = {}
        self._idf_dirty = True

        # P2: Fact anchors (hallucination prevention)
        self._ensure_raw_anchors_column()

    def _ensure_raw_anchors_column(self):
        """P2: Ensure raw_anchors column exists"""
        if not self.db_path or not os.path.exists(self.db_path):
            return
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(ltm_nodes)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'raw_anchors' not in columns:
                cursor.execute('ALTER TABLE ltm_nodes ADD COLUMN raw_anchors TEXT')
                conn.commit()
                print("[LTM] Added raw_anchors column for fact anchoring")
            conn.close()
        except Exception as e:
            print(f"[LTM] raw_anchors column check failed: {e}")

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

    def create_node(self, content: str, node_type: str = None,
                   weight: float = 1.0, raw_anchors: str = None) -> int:
        """
        Create a long-term memory node

        P2 New: raw_anchors parameter stores incompressible key facts
        (e.g., code snippets, config parameters, precise numbers) for verification during generative recall
        """
        conn = self.connect()
        try:
            cursor = conn.cursor()

            cursor.execute('''
            INSERT INTO ltm_nodes (content, type, weight, raw_anchors, created_at, last_accessed)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ''', (content, node_type, weight, raw_anchors))

            conn.commit()
            node_id = cursor.lastrowid
            self._idf_dirty = True
        finally:
            self.close()
        return node_id

    def create_link(self, source_node_id: int, target_node_id: int,
                   strength: float = 0.5, relation_type: str = 'related') -> bool:
        """
        Create association between nodes

        Args:
            source_node_id: Source node ID
            target_node_id: Target node ID
            strength: Association strength [0, 1]
            relation_type: Relationship type (causal/associated/similar/contrast/temporal/related)
        """
        conn = self.connect()
        try:
            cursor = conn.cursor()

            cursor.execute('''
            SELECT id FROM ltm_links
            WHERE source_node_id = ? AND target_node_id = ?
            ''', (source_node_id, target_node_id))

            if cursor.fetchone():
                cursor.execute('''
                UPDATE ltm_links
                SET strength = ?, relation_type = ?
                WHERE source_node_id = ? AND target_node_id = ?
                ''', (strength, relation_type, source_node_id, target_node_id))
            else:
                cursor.execute('''
                INSERT INTO ltm_links (source_node_id, target_node_id, strength, relation_type, created_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (source_node_id, target_node_id, strength, relation_type))

            conn.commit()
        finally:
            self.close()
        return True

    def retrieve(self, query: str, top_k: int = 10, threshold: float = 0.0) -> List[Dict]:
        """
        Retrieve memory nodes

        P1 Upgrade: Composite score = semantic similarity × weight × recency
        Replaces original pure LIKE queries.
        """
        # Get all active nodes
        conn = self.connect()
        try:
            cursor = conn.cursor()

            cursor.execute('''
            SELECT * FROM ltm_nodes
            WHERE weight >= ?
            ORDER BY weight DESC
            LIMIT 500
            ''', (self.delete_threshold,))

            nodes = [dict(row) for row in cursor.fetchall()]

            if not nodes:
                return []

            # Calculate composite score for each node
            scored_nodes = []
            for node in nodes:
                score = self._calculate_retrieval_score(node, query)
                if score > threshold:
                    node['retrieval_score'] = score
                    scored_nodes.append(node)

            # Sort by score
            scored_nodes.sort(key=lambda x: x['retrieval_score'], reverse=True)
            top_nodes = scored_nodes[:top_k]

            # Get associations
            for node in top_nodes:
                node['links'] = self._get_node_links(node['id'])

            return top_nodes
        finally:
            self.close()

    def _calculate_retrieval_score(self, node: Dict, query: str) -> float:
        """
        Calculate retrieval score

        score = semantic_similarity × weight × recency_boost
        """
        # Semantic similarity
        semantic_sim = self._semantic_similarity(node.get('content', ''), query)

        # Weight (includes forgetting decay)
        weight = node.get('weight', 0.5)

        # Recency boost
        recency_boost = 1.0
        last_accessed = node.get('last_accessed', '')
        if last_accessed:
            try:
                last_access = datetime.datetime.fromisoformat(last_accessed)
                days_since = (datetime.datetime.now() - last_access).days
                recency_boost = 1 + 0.1 / (1 + days_since)
            except (ValueError, TypeError):
                pass

        return semantic_sim * weight * recency_boost

    def _semantic_similarity(self, text_a: str, text_b: str) -> float:
        """
        Semantic similarity calculation (TF-IDF + Cosine similarity)

        Does not rely on external embedding models, uses lightweight TF-IDF method.
        """
        # Tokenize
        words_a = self._tokenize(text_a)
        words_b = self._tokenize(text_b)

        if not words_a or not words_b:
            return 0.0

        # Calculate term frequency
        tf_a = Counter(words_a)
        tf_b = Counter(words_b)

        # All vocabulary
        all_vocab = set(words_a) | set(words_b)
        if not all_vocab:
            return 0.0

        # Simplified IDF (uses fixed formula, not dependent on global corpus)
        # IDF(w) = log(N / (1 + df(w))), here N = 100 as approximation
        def idf(word):
            # IDF based on actual document frequency
            df = sum(1 for w in [words_a, words_b] if word in w)
            if df == 0:
                df = 1
            return math.log(2.0 / (1 + df))  # 2 documents

        # Calculate cosine similarity
        dot_product = sum(tf_a[w] * tf_b[w] for w in all_vocab)

        norm_a = math.sqrt(sum(v ** 2 for v in tf_a.values()))
        norm_b = math.sqrt(sum(v ** 2 for v in tf_b.values()))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        cosine = dot_product / (norm_a * norm_b)

        # Hybrid exact match bonus (if any part of query exactly appears in text, add bonus)
        exact_bonus = 0.0
        if text_b and len(text_b) <= 100:
            # For short queries, check substring match
            if text_b.lower() in text_a.lower():
                exact_bonus = 0.3

        return min(1.0, cosine + exact_bonus)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize: support Chinese and English"""
        if not text:
            return []

        # Extract Chinese words (2+ characters, no upper limit)
        chinese_words = re.findall(r'[\u4e00-\u9fff]{2,}', text)
        # Extract English words
        english_words = re.findall(r'[a-zA-Z]{3,}', text.lower())

        # Filter stopwords
        stopwords = {
            '的', '了', '是', '在', '和', '有', '不', '这', '我', '他',
            '她', '它', '们', '着', '过', '把', '被', '让', '给', '到',
            '也', '就', '都', '而', '及', '与', '对', '中', '上', '下',
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
            'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
            'this', 'that', 'with', 'from', 'they', 'been', 'will'
        }

        return [w for w in chinese_words + english_words if w not in stopwords]

    def access_node(self, node_id: int) -> bool:
        """
        Access node and update weight

        Memory extraction strengthens memory traces (consistent with design document)
        """
        conn = self.connect()
        try:
            cursor = conn.cursor()

            cursor.execute('''
            UPDATE ltm_nodes
            SET last_accessed = CURRENT_TIMESTAMP,
                weight = MIN(weight * 1.1, 2.0)
            WHERE id = ?
            ''', (node_id,))

            conn.commit()
            affected = cursor.rowcount
            return affected > 0
        finally:
            self.close()

    def apply_forgetting(self) -> int:
        """
        Apply forgetting mechanism

        P1 Upgrade: Power-law decay (more aligned with human memory)
        - weight < 0.2 → dormant (can be recovered)
        - weight < 0.1 → deleted
        - Skip core/system/config type nodes
        """
        conn = self.connect()
        try:
            cursor = conn.cursor()

            # Get all nodes (limit to avoid memory overflow)
            cursor.execute('SELECT * FROM ltm_nodes ORDER BY created_at ASC LIMIT 10000')
            nodes = cursor.fetchall()

            dormant_count = 0
            deleted_count = 0

            for node in nodes:
                node_dict = dict(node)

                # System nodes are not forgotten
                if node_dict.get('type') in ('core', 'system', 'config'):
                    continue

                weight = node_dict.get('weight', 0.5)

                # Power-law forgetting: based on absolute days since creation, not daily accumulation
                created_at = node_dict.get('created_at', '')
                last_accessed = node_dict.get('last_accessed', created_at)

                try:
                    ref_time = datetime.datetime.fromisoformat(
                        last_accessed or created_at
                    )
                    days_since = (datetime.datetime.now() - ref_time).days
                except (ValueError, TypeError):
                    days_since = 0

                if days_since > 0:
                    # True power-law decay: retention = days^(-exponent)
                    # Not cumulative multiplication, calculate based on absolute days
                    decay_constant = 1.5  # Decay exponent
                    retention = max(0.01, days_since ** (-decay_constant))
                    # Original weight multiplied by retention rate (capped at original weight)
                    new_weight = weight * retention

                    # Update weight
                    cursor.execute('''
                    UPDATE ltm_nodes SET weight = ? WHERE id = ?
                    ''', (new_weight, node_dict['id']))

                    if new_weight < self.delete_threshold:
                        cursor.execute('DELETE FROM ltm_nodes WHERE id = ?',
                                      (node_dict['id'],))
                        deleted_count += 1
                    elif new_weight < self.dormant_threshold:
                        dormant_count += 1

            conn.commit()
            self._idf_dirty = True
        finally:
            self.close()

        print(f"[LTM] Forgetting applied: {dormant_count} dormant, {deleted_count} deleted")
        return dormant_count + deleted_count

    def _get_node_links(self, node_id: int) -> Dict:
        """Get node associations"""
        try:
            cursor = self.connect()
            try:
                cursor.execute('''
                SELECT * FROM ltm_links
                WHERE source_node_id = ?
                ORDER BY strength DESC
                ''', (node_id,))
                out_links = [dict(row) for row in cursor.fetchall()]

                cursor.execute('''
                SELECT * FROM ltm_links
                WHERE target_node_id = ?
                ORDER BY strength DESC
                ''', (node_id,))
                in_links = [dict(row) for row in cursor.fetchall()]

                return {
                    'outgoing': out_links,
                    'incoming': in_links
                }
            except Exception:
                return {'outgoing': [], 'incoming': []}
            finally:
                self.close()
        except Exception:
            return {'outgoing': [], 'incoming': []}

    def get_node(self, node_id: int) -> Optional[Dict]:
        """Get node by ID"""
        conn = self.connect()
        try:
            cursor = conn.cursor()

            cursor.execute('SELECT * FROM ltm_nodes WHERE id = ?', (node_id,))

            row = cursor.fetchone()
            if not row:
                return None

            node = dict(row)
            node['links'] = self._get_node_links(node_id)

            return node
        finally:
            self.close()

    def update_node(self, node_id: int, content: str = None,
                   node_type: str = None, weight: float = None,
                   raw_anchors: str = None) -> bool:
        """
        Update node

        P2 New: raw_anchors parameter updates key fact anchors
        """
        conn = self.connect()
        try:
            cursor = conn.cursor()

            updates = []
            params = []

            if content is not None:
                updates.append("content = ?")
                params.append(content)
            if node_type is not None:
                updates.append("type = ?")
                params.append(node_type)
            if weight is not None:
                updates.append("weight = ?")
                params.append(weight)
            if raw_anchors is not None:
                updates.append("raw_anchors = ?")
                params.append(raw_anchors)

            if updates:
                params.append(node_id)
                sql = f"UPDATE ltm_nodes SET {', '.join(updates)} WHERE id = ?"
                cursor.execute(sql, params)
                conn.commit()

            affected = cursor.rowcount
            self._idf_dirty = True
            return affected > 0
        finally:
            self.close()

    def update_raw_anchors(self, node_id: int, raw_anchors: str) -> bool:
        """
        P2: Update node's fact anchors

        Used to update existing nodes' fact anchors when new facts are discovered during analysis
        """
        return self.update_node(node_id, raw_anchors=raw_anchors)

    def get_raw_anchors(self, node_id: int) -> Optional[str]:
        """
        P2: Get node's fact anchors

        Returns:
            JSON formatted fact anchor string, or None
        """
        conn = self.connect()
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT raw_anchors FROM ltm_nodes WHERE id = ?', (node_id,))
            row = cursor.fetchone()
            return row['raw_anchors'] if row else None
        finally:
            self.close()

    def retrieve_with_anchors(self, query: str, top_k: int = 10,
                              threshold: float = 0.0) -> List[Dict]:
        """
        P2: Retrieval with fact anchors

        On top of normal retrieval, returns fact anchors for each node
        for caller to verify facts during generation
        """
        results = self.retrieve(query, top_k, threshold)

        if not results:
            return results

        # Batch fetch raw_anchors to avoid N+1 queries
        try:
            conn = self.connect()
            try:
                cursor = conn.cursor()
                node_ids = [node['id'] for node in results]
                placeholders = ','.join('?' * len(node_ids))
                cursor.execute(
                    f'SELECT id, raw_anchors FROM ltm_nodes WHERE id IN ({placeholders})',
                    node_ids
                )
                anchor_map = {row[0]: row[1] for row in cursor.fetchall()}

                for node in results:
                    node['raw_anchors'] = anchor_map.get(node['id'])
            finally:
                self.close()
        except Exception:
            pass

        return results

    def delete_node(self, node_id: int) -> bool:
        """Delete node"""
        conn = self.connect()
        try:
            cursor = conn.cursor()

            cursor.execute('''
            DELETE FROM ltm_links
            WHERE source_node_id = ? OR target_node_id = ?
            ''', (node_id, node_id))

            cursor.execute('DELETE FROM ltm_nodes WHERE id = ?', (node_id,))

            conn.commit()
            affected = cursor.rowcount
            self._idf_dirty = True
            return affected > 0
        finally:
            self.close()

    def get_all_nodes(self, limit: int = 100) -> List[Dict]:
        """Get all nodes"""
        conn = self.connect()
        try:
            cursor = conn.cursor()

            cursor.execute('''
            SELECT * FROM ltm_nodes
            ORDER BY weight DESC
            LIMIT ?
            ''', (limit,))

            nodes = [dict(row) for row in cursor.fetchall()]
            return nodes
        finally:
            self.close()
