"""回归测试脚本 - PersonaMem v3 (路线A)"""

import os
import sys
import unittest

# 添加项目根目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.schema import Schema
from core.working_memory import WorkingMemory
from core.long_term_memory import LongTermMemory
from core.persona_profile import PersonaProfile

class TestRegression(unittest.TestCase):
    def setUp(self):
        """设置测试环境"""
        self.test_db = 'test_regression.db'
        self.schema = Schema(self.test_db)
        self.schema.init_db()
        
        self.working_memory = WorkingMemory(self.test_db)
        self.long_term_memory = LongTermMemory(self.test_db)
        self.persona_profile = PersonaProfile(self.test_db)
    
    def tearDown(self):
        """清理测试环境"""
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
    
    def test_working_memory_regression(self):
        """测试工作记忆回归"""
        # 测试边界情况
        # 空内容
        entry_id = self.working_memory.capture("")
        self.assertIsNotNone(entry_id)
        
        # 长内容
        long_content = "a" * 10000
        entry_id = self.working_memory.capture(long_content)
        self.assertIsNotNone(entry_id)
        
        # 测试异常情况
        # 无效的entry_id
        result = self.working_memory.mark_consolidated(999999)
        self.assertFalse(result)
        
        # 测试过期清理
        deleted = self.working_memory.expire_old_entries()
        self.assertGreaterEqual(deleted, 0)
    
    def test_long_term_memory_regression(self):
        """测试长期记忆回归"""
        # 测试边界情况
        # 空内容
        node_id = self.long_term_memory.create_node("")
        self.assertIsNotNone(node_id)
        
        # 测试异常情况
        # 无效的node_id
        result = self.long_term_memory.access_node(999999)
        self.assertFalse(result)
        
        # 无效的关联
        result = self.long_term_memory.create_link(999999, 999999)
        self.assertTrue(result)  # 应该创建成功，但不会有实际效果
        
        # 测试遗忘机制
        forgotten = self.long_term_memory.apply_forgetting()
        self.assertGreaterEqual(forgotten, 0)
    
    def test_persona_profile_regression(self):
        """测试 PersonaProfile 记忆痕迹系统回归"""
        # 测试边界情况
        # 空内容
        trace_id = self.persona_profile.store_experience("")
        self.assertIsNotNone(trace_id)
        
        # 测试异常情况
        # 无效的trace_id
        trace = self.persona_profile.get_trace(999999)
        self.assertIsNone(trace)
        
        # 测试空条件检索
        traces = self.persona_profile.retrieve_by_conditions([])
        self.assertEqual(len(traces), 0)

if __name__ == '__main__':
    unittest.main()