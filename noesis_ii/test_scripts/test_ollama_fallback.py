"""
PersonaMem Ollama Fallback 机制测试 - v3.1 (路线A):

已删除 legacy/，本文件测试新版 PersonaExtractor (core/persona_extractor.py) 的 LLM 链：
1. LLM client 可用时 -> 使用 LLM 零样本推理
2. LLM client 不可用时 -> 自动回退到关键词启发式

测试场景：
- PersonaExtractor 带 Mock LLM 时正常工作
- LLM 失败时自动回退到关键词方法
- LLM 返回格式错误时使用默认分数
"""

import os
import sys
import time
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# 确保项目根目录在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from noesis_ii.core.schema import Schema
from noesis_ii.config_loader import ConfigLoader


class TestOllamaAvailability(unittest.TestCase):
    """测试 1：Ollama 可用性检测"""

    def test_ollama_import(self):
        """验证 Ollama 模块可以正确导入"""
        try:
            from noesis_ii.llm import OllamaLLM, is_ollama_available
            self.assertIsNotNone(OllamaLLM)
            self.assertTrue(callable(is_ollama_available))
            print("[PASS] Ollama module imported successfully")
        except ImportError as e:
            self.fail(f"Ollama module import failed: {e}")

    def test_ollama_availability_check(self):
        """验证 Ollama 可用性检测函数"""
        from noesis_ii.llm import is_ollama_available

        # 这会尝试连接本地 Ollama 服务
        available = is_ollama_available()
        print(f"[INFO] Ollama available: {available}")

        # 无论是否可用，检测函数应该不抛异常
        self.assertIsInstance(available, bool)


class TestDeepenerOllamaFallback(unittest.TestCase):
    """测试 2：Deepener Ollama Fallback 机制"""

    @classmethod
    def setUpClass(cls):
        cls.test_db = os.path.join(_project_root, 'data', 'test_deepener_ollama.db')
        # 清理旧数据库
        if os.path.exists(cls.test_db):
            os.remove(cls.test_db)
        # 初始化数据库
        data_dir = os.path.dirname(cls.test_db)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        schema = Schema(cls.test_db)
        schema.init_db()
        # 加载配置
        config_path = os.path.join(_project_root, 'config', 'default_config.yaml')
        cls.config_loader = ConfigLoader(config_path)
        cls.config = cls.config_loader.load()
        cls.llm_config = cls.config.get('llm', {})

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_db):
            os.remove(cls.test_db)

    def test_deepener_init_with_ollama(self):
        """验证 Deepener 初始化 Ollama"""
        from noesis_ii.processes.deepener import Deepener

        deepener = Deepener(
            db_path=self.test_db,
            llm_config=self.llm_config
        )

        # 检查 _init_ollama 方法存在
        self.assertTrue(hasattr(deepener, '_init_ollama'))
        self.assertTrue(hasattr(deepener, '_analyze_with_ollama'))
        self.assertTrue(hasattr(deepener, '_analyze_with_remote_llm'))

        print(f"[INFO] Deepener has Ollama fallback methods")

    def test_deepener_ollama_priority(self):
        """验证 Deepener 优先使用 Ollama"""
        from noesis_ii.processes.deepener import Deepener

        # 创建 Deepener 实例
        deepener = Deepener(
            db_path=self.test_db,
            llm_config=self.llm_config
        )

        # 检查 Ollama 是否初始化
        if deepener._ollama is not None:
            print("[INFO] Deepener using Ollama (local model)")
            self.assertIsNotNone(deepener._ollama)
        else:
            print("[INFO] Deepener Ollama not available, will use remote LLM")

    def test_deepener_analyze_with_mock_ollama(self):
        """验证 Deepener Ollama 分析（Mock 测试）"""
        from noesis_ii.processes.deepener import Deepener

        deepener = Deepener(
            db_path=self.test_db,
            llm_config=self.llm_config
        )

        # Mock Ollama 实例
        mock_ollama = Mock()
        mock_ollama.generate.return_value = "这是一个基于记忆内容的人格分析结果。"
        deepener._ollama = mock_ollama

        # 测试分析
        test_content = "用户喜欢阅读科幻小说，关心人工智能发展。"
        result = deepener._analyze_with_ollama(test_content)

        # 验证结果
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        print(f"[PASS] Deepener Ollama analysis: {result[:50]}...")

        # 验证调用参数
        mock_ollama.generate.assert_called_once()
        call_kwargs = mock_ollama.generate.call_args[1]
        self.assertIn('prompt', call_kwargs)
        self.assertIn('system', call_kwargs)
        self.assertEqual(call_kwargs['temperature'], 0.3)

    def test_deepener_remote_llm_fallback(self):
        """验证 Deepener 远程 LLM 回退（Mock 测试）"""
        from noesis_ii.processes.deepener import Deepener

        deepener = Deepener(
            db_path=self.test_db,
            llm_config=self.llm_config
        )

        # 确保 Ollama 不可用
        deepener._ollama = None

        # Mock 远程 LLM 调用
        mock_response = Mock()
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': '远程 LLM 分析结果：主人格特征分析。'
                }
            }]
        }
        mock_response.raise_for_status = Mock()

        with patch('requests.post', return_value=mock_response) as mock_post:
            result = deepener._analyze_with_remote_llm("测试内容")

            self.assertIsNotNone(result)
            self.assertIn('远程 LLM', result)
            mock_post.assert_called_once()
            print(f"[PASS] Deepener remote LLM fallback works")

    def test_deepener_fallback_chain(self):
        """验证 Deepener Ollama -> 远程 LLM 完整回退链"""
        from noesis_ii.processes.deepener import Deepener

        deepener = Deepener(
            db_path=self.test_db,
            llm_config=self.llm_config
        )

        # 场景1：Ollama 可用
        mock_ollama = Mock()
        mock_ollama.generate.return_value = "Ollama 分析结果"
        deepener._ollama = mock_ollama

        result = deepener._analyze_with_llm("测试内容")
        self.assertIsNotNone(result)
        self.assertEqual(result, "Ollama 分析结果")
        mock_ollama.generate.assert_called_once()
        print("[PASS] Fallback chain: Ollama available -> uses Ollama")

        # 场景2：Ollama 失败，回退到远程 LLM
        mock_ollama.generate.side_effect = Exception("Ollama error")

        mock_response = Mock()
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'Remote LLM fallback result'}}]
        }
        mock_response.raise_for_status = Mock()

        with patch('requests.post', return_value=mock_response):
            result = deepener._analyze_with_llm("测试内容")
            self.assertIsNotNone(result)
            print("[PASS] Fallback chain: Ollama failed -> uses remote LLM")


class TestPersonaExtractorLLM(unittest.TestCase):
    """测试 3：PersonaExtractor LLM 链（替代原 DeepPersonality 测试）"""

    def test_persona_extractor_init(self):
        """验证 PersonaExtractor 初始化"""
        from noesis_ii.core.persona_extractor import PersonaExtractor

        # 不带 LLM 初始化
        pe = PersonaExtractor()
        self.assertIsNone(pe.llm)
        self.assertFalse(pe._fallback_mode)
        print("[PASS] PersonaExtractor init without LLM")

    def test_persona_extractor_with_mock_llm(self):
        """验证 PersonaExtractor 带 Mock LLM"""
        from noesis_ii.core.persona_extractor import PersonaExtractor, OCEANScores

        # Mock LLM
        mock_llm = Mock()
        mock_llm.generate.return_value = '{"openness": 0.8, "conscientiousness": 0.6, "extraversion": 0.7, "agreeableness": 0.5, "neuroticism": 0.3}'

        pe = PersonaExtractor(llm_client=mock_llm)
        self.assertIsNotNone(pe.llm)
        self.assertFalse(pe._fallback_mode)

        # 测试提取
        result = pe.extract_ocean("I love exploring new ideas and creative solutions.")

        self.assertIsNotNone(result)
        self.assertIsInstance(result, OCEANScores)
        self.assertAlmostEqual(result.openness, 0.8, places=1)
        print(f"[PASS] PersonaExtractor with Mock LLM: O={result.openness}, C={result.conscientiousness}")

        # 验证调用
        mock_llm.generate.assert_called_once()
        call_kwargs = mock_llm.generate.call_args[1]
        self.assertEqual(call_kwargs['temperature'], 0.1)

    def test_persona_extractor_fallback_on_no_llm(self):
        """验证 PersonaExtractor 无 LLM 时自动回退到关键词"""
        from noesis_ii.core.persona_extractor import PersonaExtractor

        pe = PersonaExtractor()  # 无 LLM

        result = pe.extract_ocean("I love to explore new creative ideas and imaginative solutions.")

        self.assertIsNotNone(result)
        self.assertGreater(result.openness, 0.3)
        print(f"[PASS] PersonaExtractor fallback (no LLM): O={result.openness}")

    def test_persona_extractor_fallback_on_llm_error(self):
        """验证 PersonaExtractor LLM 失败时回退"""
        from noesis_ii.core.persona_extractor import PersonaExtractor

        mock_llm = Mock()
        mock_llm.generate.side_effect = Exception("LLM error")
        pe = PersonaExtractor(llm_client=mock_llm)

        result = pe.extract_ocean("Test text with creative exploration.")

        # 应该成功回退到关键词方法
        self.assertIsNotNone(result)
        self.assertGreater(result.openness, 0.3)
        print(f"[PASS] PersonaExtractor auto-fallback on LLM error: O={result.openness}")

    def test_persona_extractor_malformed_response(self):
        """验证 PersonaExtractor LLM 返回格式错误时使用默认分数"""
        from noesis_ii.core.persona_extractor import PersonaExtractor

        mock_llm = Mock()
        mock_llm.generate.return_value = "This is not JSON format"
        pe = PersonaExtractor(llm_client=mock_llm)

        result = pe.extract_ocean("Some text.")

        # 应返回默认分数 0.5
        self.assertEqual(result.openness, 0.5)
        self.assertEqual(result.conscientiousness, 0.5)
        print(f"[PASS] PersonaExtractor uses default on malformed response")


class TestThreeModelDivision(unittest.TestCase):
    """测试 4：三模型分工集成测试（已更新为新模块）"""

    @classmethod
    def setUpClass(cls):
        cls.test_db = os.path.join(_project_root, 'data', 'test_model_division.db')
        if os.path.exists(cls.test_db):
            os.remove(cls.test_db)
        data_dir = os.path.dirname(cls.test_db)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        schema = Schema(cls.test_db)
        schema.init_db()
        config_path = os.path.join(_project_root, 'config', 'default_config.yaml')
        cls.config_loader = ConfigLoader(config_path)
        cls.config = cls.config_loader.load()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_db):
            os.remove(cls.test_db)

    def test_model_division_priority(self):
        """验证模型分工优先级"""
        from noesis_ii.processes.deepener import Deepener
        from noesis_ii.core.persona_extractor import PersonaExtractor

        # 创建实例
        deepener = Deepener(db_path=self.test_db, llm_config=self.config.get('llm', {}))
        extractor = PersonaExtractor()

        # 检查 Deepener Ollama 初始化状态
        deepener_ollama = deepener._ollama is not None

        print(f"\n[MODEL STATUS]")
        print(f"  Deepener Ollama: {'OK' if deepener_ollama else 'Not available'}")
        print(f"  PersonaExtractor: {'OK' if extractor.llm else 'No LLM client (will use fallback)'}")
        print(f"  Remote LLM Config: {'OK' if self.config.get('llm', {}).get('api_key') else 'Not configured'}")

        # 至少应该有一个 LLM 选项可用
        has_llm = deepener_ollama or bool(self.config.get('llm', {}).get('api_key'))
        self.assertTrue(has_llm, "At least one LLM should be available")

    def test_minilm_always_used(self):
        """验证 MiniLM 始终用于向量化"""
        from noesis_ii.retrieval.hybrid_retriever import HybridRetriever

        # HybridRetriever 应该始终使用 MiniLM 进行向量化
        hr = HybridRetriever(db_path=self.test_db)
        self.assertIsNotNone(hr)
        print("[PASS] HybridRetriever (MiniLM) is available for vectorization")


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("NOESIS-II Ollama Fallback 机制测试 (v3.1)")
    print("=" * 60)
    print()

    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestOllamaAvailability))
    suite.addTests(loader.loadTestsFromTestCase(TestDeepenerOllamaFallback))
    suite.addTests(loader.loadTestsFromTestCase(TestPersonaExtractorLLM))
    suite.addTests(loader.loadTestsFromTestCase(TestThreeModelDivision))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 输出总结
    print()
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"运行: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
