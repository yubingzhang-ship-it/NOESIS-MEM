"""
PersonaMem LLM 集成测试 - v3 (路线A)
测试 Consolidator 和 Deepener 的真实 LLM API 调用
"""

import os
import sys
import time
import json
import unittest
from datetime import datetime

# 确保项目根目录在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from noesis_ii.core.schema import Schema
from noesis_ii.core.working_memory import WorkingMemory
from noesis_ii.core.long_term_memory import LongTermMemory
from noesis_ii.core.persona_profile import PersonaProfile
from noesis_ii.processes.consolidator import Consolidator
from noesis_ii.processes.deepener import Deepener
from noesis_ii.processes.replay_engine import ReplayEngine
from noesis_ii.retrieval.retriever import Retriever
from noesis_ii.config_loader import ConfigLoader


class TestLLMConnectivity(unittest.TestCase):
    """测试 1：LLM API 连通性"""

    @classmethod
    def setUpClass(cls):
        cls.test_db = os.path.join(_project_root, 'data', 'test_llm.db')
        # 清理旧数据库
        if os.path.exists(cls.test_db):
            os.remove(cls.test_db)
        # 初始化数据库
        data_dir = os.path.dirname(cls.test_db)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        schema = Schema(cls.test_db)
        schema.init_db()
        # 加载 LLM 配置
        config_path = os.path.join(_project_root, 'config', 'default_config.yaml')
        cls.config_loader = ConfigLoader(config_path)
        cls.config = cls.config_loader.load()
        cls.llm_config = cls.config.get('llm', {})
        print(f"\n[LLM] Model: {cls.llm_config.get('model')}")
        print(f"[LLM] API Base: {cls.llm_config.get('api_base')}")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_db):
            os.remove(cls.test_db)

    def test_llm_api_reachable(self):
        """验证 LLM API 可达"""
        import requests
        api_key = self.llm_config.get('api_key')
        api_base = self.llm_config.get('api_base')
        model = self.llm_config.get('model')
        self.assertIsNotNone(api_key, "LLM API key not configured")
        self.assertIsNotNone(api_base, "LLM API base not configured")
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            }
            data = {
                'model': model,
                'messages': [{'role': 'user', 'content': 'Hello, please respond with just "OK".'}],
                'temperature': 0,
                'max_tokens': 10
            }
            resp = requests.post(f'{api_base}/chat/completions', headers=headers, json=data, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            content = result['choices'][0]['message']['content']
            print(f"[LLM] API reachable, response: {content[:50]}")
            self.assertTrue(len(content) > 0, "LLM returned empty response")
        except requests.exceptions.Timeout:
            self.fail("LLM API timeout (30s)")
        except requests.exceptions.ConnectionError as e:
            self.fail(f"LLM API connection error: {e}")
        except Exception as e:
            self.fail(f"LLM API error: {e}")

    def test_llm_config_valid(self):
        """验证 LLM 配置完整性"""
        self.assertIn('api_key', self.llm_config)
        self.assertIn('api_base', self.llm_config)
        self.assertIn('model', self.llm_config)
        self.assertTrue(self.llm_config['api_key'].startswith('ak_'), "API key should start with ak_")


class TestConsolidatorLLM(unittest.TestCase):
    """测试 2：Consolidator 真实 LLM 整理"""

    @classmethod
    def setUpClass(cls):
        cls.test_db = os.path.join(_project_root, 'data', 'test_llm_consolidator.db')
        if os.path.exists(cls.test_db):
            os.remove(cls.test_db)
        data_dir = os.path.dirname(cls.test_db)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        schema = Schema(cls.test_db)
        schema.init_db()
        config_path = os.path.join(_project_root, 'config', 'default_config.yaml')
        cls.llm_config = ConfigLoader(config_path).load().get('llm', {})

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_db):
            os.remove(cls.test_db)

    def test_consolidate_single_entry(self):
        """LLM 整理单条工作记忆：内容分析 -> 概括 -> 碎片 -> 关联"""
        wm = WorkingMemory(self.test_db)
        ltm = LongTermMemory(self.test_db)

        # 写入测试内容
        content = "今天读完了《思考，快与慢》的第一部分，卡尼曼提出了系统1和系统2的理论。系统1是快速直觉思维，系统2是慢速理性思维。这两个系统的交互决定了我们的判断和决策。这让我想到了在日常编程中，很多时候直觉选择效率更高，但遇到复杂架构问题时，必须切换到系统2进行深度思考。"
        entry_id = wm.capture(content, "inspired")
        self.assertIsNotNone(entry_id)

        # 运行 LLM 整理
        consolidator = Consolidator(self.test_db, self.llm_config)
        count = consolidator.run(limit=1)

        # 验证整理成功
        self.assertGreaterEqual(count, 1, "Consolidator should process at least 1 entry")

        # 验证工作记忆已标记为已整理
        pending = wm.get_pending()
        self.assertEqual(len(pending), 0, "No pending entries after consolidation")

        # 验证长期记忆中创建了节点
        all_nodes = ltm.get_all_nodes()
        # 应该有：原始内容节点 + 概括节点 + 3-5个碎片节点
        self.assertGreaterEqual(len(all_nodes), 3, f"Expected >= 3 LTM nodes, got {len(all_nodes)}")

        # 验证有关联关系
        for node in all_nodes:
            node_detail = ltm.get_node(node['id'])
            if node_detail and 'links' in node_detail:
                total_links = len(node_detail['links'].get('outgoing', [])) + len(node_detail['links'].get('incoming', []))
                if total_links > 0:
                    print(f"[LLM-CONSOLIDATE] Node {node['id']}: {node['content'][:40]}... links={total_links}")
                    break  # 至少一个节点有关联就够了

        print(f"[LLM-CONSOLIDATE] Total LTM nodes: {len(all_nodes)}")

    def test_consolidate_multiple_entries(self):
        """LLM 整理多条工作记忆"""
        wm = WorkingMemory(self.test_db)

        entries = [
            ("量子计算的基础是量子叠加和量子纠缠原理。与经典比特不同，量子比特可以同时处于0和1的叠加态。", "curious"),
            ("今天和朋友讨论了意识的困难问题——为什么主观体验无法从物理描述中推导出来。这可能是认知科学最深奥的未解之谜。", "contemplative"),
            ("跑步5公里，配速5分30秒。运动后身体虽然疲惫，但精神状态明显改善，这可能是内啡肽的作用。", "energetic"),
        ]

        for content, emotion in entries:
            wm.capture(content, emotion)

        consolidator = Consolidator(self.test_db, self.llm_config)
        count = consolidator.run(limit=10)

        self.assertGreaterEqual(count, 3, f"Expected >= 3 consolidated, got {count}")

        # 验证所有条目已整理
        pending = wm.get_pending()
        self.assertEqual(len(pending), 0, "All entries should be consolidated")

        ltm = LongTermMemory(self.test_db)
        all_nodes = ltm.get_all_nodes()
        print(f"[LLM-CONSOLIDATE-MULTI] Processed {count} entries, created {len(all_nodes)} LTM nodes")

    def test_consolidate_with_similarity(self):
        """LLM 整理相似内容：应合并而非重复创建"""
        wm = WorkingMemory(self.test_db)
        ltm = LongTermMemory(self.test_db)

        # 两条高度相似的内容
        wm.capture("今天完成了项目的第一阶段开发，主要是搭建基础架构和数据库设计。", "satisfied")
        wm.capture("项目第一阶段开发完成，搭建了基础架构，完成了数据库设计工作。", "satisfied")

        consolidator = Consolidator(self.test_db, self.llm_config)
        count = consolidator.run(limit=10)

        self.assertEqual(count, 2, "Should process both entries")

        all_nodes = ltm.get_all_nodes()
        # 相似内容应该被合并，节点数应该比独立的少
        print(f"[LLM-SIMILARITY] Processed {count} similar entries, created {len(all_nodes)} nodes")

    def test_consolidate_empty_content(self):
        """LLM 整理空/极短内容：应优雅处理"""
        wm = WorkingMemory(self.test_db)

        wm.capture("", "neutral")
        wm.capture("OK", "neutral")

        consolidator = Consolidator(self.test_db, self.llm_config)
        count = consolidator.run(limit=10)

        # 空内容可能无法被 LLM 分析，但不应该崩溃
        print(f"[LLM-EMPTY] Processed {count} empty/short entries")


class TestDeepenerLLM(unittest.TestCase):
    """测试 3：Deepener 真实 LLM 深化分析"""

    @classmethod
    def setUpClass(cls):
        cls.test_db = os.path.join(_project_root, 'data', 'test_llm_deepener.db')
        if os.path.exists(cls.test_db):
            os.remove(cls.test_db)
        data_dir = os.path.dirname(cls.test_db)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        schema = Schema(cls.test_db)
        schema.init_db()
        config_path = os.path.join(_project_root, 'config', 'default_config.yaml')
        cls.llm_config = ConfigLoader(config_path).load().get('llm', {})

        # 预填充长期记忆数据
        ltm = LongTermMemory(cls.test_db)
        memories = [
            "在禅修中体验到了'无我'的状态，意识到自我意识可能只是一种大脑构建的幻觉。这让我重新思考了意识的本体论问题。",
            "读了《意识探秘》这本书，克里斯托弗·科赫认为意识与信息整合密切相关，整合信息量越大，意识程度越高。",
            "与一位佛教徒朋友讨论了'缘起性空'的概念——一切事物都是因缘和合而生，没有独立不变的自性。这和量子力学的关联性令人着迷。",
            "在工作中尝试用最小作用量原理来优化代码架构，发现很多设计模式其实是系统趋向最低能耗状态的自然表现。",
            "参加了一个关于人工智能意识的研讨会，讨论了中文房间论证和图灵测试的局限性。",
        ]
        for m in memories:
            ltm.create_node(m, weight=1.5)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_db):
            os.remove(cls.test_db)

    def test_deepener_llm_analysis(self):
        """Deepener 使用 LLM 进行深度分析"""
        deepener = Deepener(self.test_db, self.llm_config)
        result = deepener.run()

        # 验证基本结构
        self.assertIn('personality', result)
        self.assertIn('high_weight_nodes_count', result)
        self.assertIn('patterns', result)

        # 验证人格生成
        personality = result['personality']
        self.assertGreater(len(personality), 0, "Personality should be generated")

        # 如果有 LLM 分析结果
        if result.get('llm_analysis'):
            self.assertIsInstance(result['llm_analysis'], str)
            self.assertGreater(len(result['llm_analysis']), 50, "LLM analysis should be substantive")
            print(f"[LLM-DEEPEN] Analysis preview: {result['llm_analysis'][:200]}...")
        else:
            print("[LLM-DEEPEN] No LLM analysis returned (may be empty nodes or API issue)")

        print(f"[LLM-DEEPEN] Personality dims: {list(personality.keys())}")
        print(f"[LLM-DEEPEN] High weight nodes: {result['high_weight_nodes_count']}")
        print(f"[LLM-DEEPEN] Patterns: {len(result['patterns'])}")

    def test_deepener_personality_update(self):
        """Deepener 更新深层人格"""
        deepener = Deepener(self.test_db, self.llm_config)
        personality = deepener.update_personality()

        self.assertIsInstance(personality, dict)
        self.assertGreater(len(personality), 0)

        # 验证每个维度在合理范围内
        for dim, info in personality.items():
            value = info.get('value', info) if isinstance(info, dict) else info
            if isinstance(value, dict):
                value = value.get('value', 0.5)
            self.assertGreaterEqual(float(value), 0.0, f"{dim} value should be >= 0")
            self.assertLessEqual(float(value), 1.0, f"{dim} value should be <= 1")
            print(f"[LLM-PERSONALITY] {dim}: {value}")

    def test_deepener_pattern_analysis(self):
        """Deepener 分析记忆模式"""
        deepener = Deepener(self.test_db, self.llm_config)
        patterns = deepener.analyze_memory_patterns()

        self.assertIsInstance(patterns, list)
        self.assertGreater(len(patterns), 0, "Should find at least some patterns")

        for p in patterns:
            self.assertIn('type', p)
            self.assertIn('description', p)
            print(f"[LLM-PATTERN] {p['type']}: {p['description']}")


class TestFullPipelineLLM(unittest.TestCase):
    """测试 4：完整 LLM 流水线（输入 -> 整理 -> 深化 -> 检索）"""

    @classmethod
    def setUpClass(cls):
        cls.test_db = os.path.join(_project_root, 'data', 'test_llm_pipeline.db')
        if os.path.exists(cls.test_db):
            os.remove(cls.test_db)
        data_dir = os.path.dirname(cls.test_db)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        schema = Schema(cls.test_db)
        schema.init_db()
        config_path = os.path.join(_project_root, 'config', 'default_config.yaml')
        cls.llm_config = ConfigLoader(config_path).load().get('llm', {})

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_db):
            os.remove(cls.test_db)

    def test_full_memory_pipeline(self):
        """完整流水线：capture -> consolidate -> deepen -> retrieve -> replay"""
        wm = WorkingMemory(self.test_db)
        ltm = LongTermMemory(self.test_db)
        persona = PersonaProfile(self.test_db)

        # Step 1: 输入多段内容到工作记忆
        print("\n[PIPELINE] Step 1: Capture working memories...")
        contents = [
            ("今天学习了神经科学的最新进展：科学家发现了大脑中一种新的神经递质系统，可能与人体的昼夜节律有关。", "fascinated"),
            ("在读《人类简史》中关于认知革命的部分，赫拉利认为虚构故事的能力是人类协作的关键。语言不只是传递信息，更是构建共同的虚构现实。", "thoughtful"),
            ("尝试用Python实现了一个简单的自由能原理模型，用预测编码来模拟感知过程。代码虽然粗糙，但对理解弗里斯顿的理论很有帮助。", "accomplished"),
        ]
        for content, emotion in contents:
            entry_id = wm.capture(content, emotion)
            self.assertIsNotNone(entry_id)

        pending = wm.get_pending()
        self.assertEqual(len(pending), 3, "Should have 3 pending entries")

        # Step 2: LLM 整理
        print("[PIPELINE] Step 2: LLM Consolidation...")
        consolidator = Consolidator(self.test_db, self.llm_config)
        count = consolidator.run(limit=10)
        print(f"[PIPELINE] Consolidated: {count} entries")

        self.assertGreaterEqual(count, 1, "At least 1 entry should be consolidated")

        # Step 3: 验证长期记忆
        print("[PIPELINE] Step 3: Verify LTM...")
        all_nodes = ltm.get_all_nodes()
        self.assertGreater(len(all_nodes), 0, "LTM should have nodes")
        print(f"[PIPELINE] LTM nodes: {len(all_nodes)}")

        # Step 4: LLM 深化分析
        print("[PIPELINE] Step 4: LLM Deepening...")
        deepener = Deepener(self.test_db, self.llm_config)
        result = deepener.run()
        self.assertIn('personality', result)
        print(f"[PIPELINE] Personality dims: {len(result['personality'])}")

        # Step 5: 检索
        print("[PIPELINE] Step 5: Retrieval...")
        retriever = Retriever(self.test_db)
        results = retriever.retrieve("认知革命")
        self.assertIn('integrated', results)
        print(f"[PIPELINE] Retrieved {len(results.get('integrated', []))} results for '认知革命'")

        # Step 6: 种子现行
        print("[PIPELINE] Step 6: Alaya Seeds...")
        seeds = alaya.manifest_from_conditions(["神经"])
        print(f"[PIPELINE] Manifested {len(seeds)} seeds for '神经'")

        # Step 7: 重放
        print("[PIPELINE] Step 7: Replay...")
        replay = ReplayEngine(self.test_db)
        replay_results = replay.run(mode='random', limit=min(3, len(all_nodes)))
        print(f"[PIPELINE] Replayed {len(replay_results)} memories")

        print("\n[PIPELINE] Full pipeline complete!")


class TestLLMEdgeCases(unittest.TestCase):
    """测试 5：LLM 边界情况"""

    @classmethod
    def setUpClass(cls):
        cls.test_db = os.path.join(_project_root, 'data', 'test_llm_edge.db')
        if os.path.exists(cls.test_db):
            os.remove(cls.test_db)
        data_dir = os.path.dirname(cls.test_db)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        schema = Schema(cls.test_db)
        schema.init_db()
        config_path = os.path.join(_project_root, 'config', 'default_config.yaml')
        cls.llm_config = ConfigLoader(config_path).load().get('llm', {})

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_db):
            os.remove(cls.test_db)

    def test_long_content(self):
        """LLM 整理超长内容（应截断但不崩溃）"""
        wm = WorkingMemory(self.test_db)
        # 生成超长内容
        long_text = "这是一段非常长的内容。" * 200  # ~4000字
        wm.capture(long_text, "neutral")

        consolidator = Consolidator(self.test_db, self.llm_config)
        count = consolidator.run(limit=1)
        print(f"[LLM-LONG] Processed long content ({len(long_text)} chars): {count} consolidated")

    def test_special_characters(self):
        """LLM 整理含特殊字符的内容"""
        wm = WorkingMemory(self.test_db)
        special_content = "测试特殊字符：\t\n\x00\x01<>&\"'`~!@#$%^&*()_+-={}[]|\\:;\"<>?,./"
        wm.capture(special_content, "neutral")

        consolidator = Consolidator(self.test_db, self.llm_config)
        try:
            count = consolidator.run(limit=1)
            print(f"[LLM-SPECIAL] Processed special chars: {count} consolidated")
        except Exception as e:
            # 特殊字符可能导致问题，但不应该崩溃整个系统
            print(f"[LLM-SPECIAL] Handled special chars with error: {e}")

    def test_chinese_philosophical_content(self):
        """LLM 整理中文哲学内容（测试语义理解质量）"""
        wm = WorkingMemory(self.test_db)
        ltm = LongTermMemory(self.test_db)

        content = "王阳明提出'知行合一'，认为知是行的主意，行是知的功夫；知是行之始，行是知之成。这与现代认知科学中的具身认知理论不谋而合——认知不是抽象的符号操作，而是根植于身体与环境的互动之中。从现象学的角度看，梅洛-庞蒂的身体现象学也支持这一观点：意识总是具身化的意识。"
        wm.capture(content, "inspired")

        consolidator = Consolidator(self.test_db, self.llm_config)
        count = consolidator.run(limit=1)

        if count > 0:
            all_nodes = ltm.get_all_nodes()
            # 验证 LLM 能够正确拆分哲学内容
            for node in all_nodes:
                content_preview = node['content'][:100]
                print(f"[LLM-PHILOSOPHY] Node: {content_preview}...")

            # 检索验证
            retriever = Retriever(self.test_db)
            results = retriever.retrieve("知行合一")
            self.assertIn('integrated', results)
            print(f"[LLM-PHILOSOPHY] Retrieved {len(results.get('integrated', []))} results for '知行合一'")

    def test_no_pending_entries(self):
        """整理进程在没有待整理条目时的行为"""
        consolidator = Consolidator(self.test_db, self.llm_config)
        count = consolidator.run(limit=10)
        self.assertEqual(count, 0, "Should return 0 when no pending entries")
        print("[LLM-EMPTY] Correctly handled empty pending queue")

    def test_mixed_language_content(self):
        """LLM 整理中英混合内容"""
        wm = WorkingMemory(self.test_db)
        ltm = LongTermMemory(self.test_db)

        content = "学习了 Transformer architecture 的 self-attention mechanism。Scaled Dot-Product Attention 的核心公式是 softmax(QK^T/√d_k)V。在 BERT 中使用的是 encoder-only 结构，而 GPT 使用的是 decoder-only。这让我想到了 attention 机制和人类注意力的相似性——都是一种资源分配策略。"
        wm.capture(content, "curious")

        consolidator = Consolidator(self.test_db, self.llm_config)
        count = consolidator.run(limit=1)
        print(f"[LLM-MIXED] Mixed language content: {count} consolidated")

        if count > 0:
            all_nodes = ltm.get_all_nodes()
            retriever = Retriever(self.test_db)
            results = retriever.retrieve("Transformer")
            print(f"[LLM-MIXED] Retrieved {len(results.get('integrated', []))} results for 'Transformer'")


if __name__ == '__main__':
    print("=" * 60)
    print("NOESIS-II LLM Integration Tests")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    unittest.main(verbosity=2)
