"""性能测试脚本 - PersonaMem v3 (路线A)

修订历史：
  v1.1 (2026-04-08) - 修复 test_working_memory_performance
  v2.0 (2026-04-10) - 路线A重构：术语替换，AlayaSeeds→PersonaProfile
"""

import os
import sys
import time
import unittest

# 添加项目根目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.schema import Schema
from core.working_memory import WorkingMemory
from core.long_term_memory import LongTermMemory
from core.persona_profile import PersonaProfile

class TestPerformance(unittest.TestCase):
    def setUp(self):
        """设置测试环境"""
        self.test_db = 'test_performance.db'
        self.schema = Schema(self.test_db)
        self.schema.init_db()
        
        self.working_memory = WorkingMemory(self.test_db)
        self.long_term_memory = LongTermMemory(self.test_db)
        self.persona_profile = PersonaProfile(self.test_db)
    
    def tearDown(self):
        """清理测试环境"""
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
    
    def test_working_memory_performance(self):
        """测试工作记忆性能"""
        # 测试工作记忆写入性能
        start_time = time.time()
        for i in range(100):
            self.working_memory.capture(f"测试经验{i}", "neutral")
        end_time = time.time()
        avg_time = (end_time - start_time) / 100 * 1000  # 转换为毫秒
        print(f"工作记忆写入平均时间: {avg_time:.2f} ms")
        # 每次 capture 含 SQLite open/commit/close 完整周期，Windows 上约 1-5ms/次
        # 阈值设为 150ms（留有合理余量）
        self.assertLess(avg_time, 150, "工作记忆写入时间应小于150ms")
        
        # 测试工作记忆读取性能
        start_time = time.time()
        entries = self.working_memory.get_all()
        end_time = time.time()
        read_time = (end_time - start_time) * 1000  # 转换为毫秒
        print(f"工作记忆读取时间: {read_time:.2f} ms")
        self.assertLess(read_time, 50, "工作记忆读取时间应小于50ms")
    
    def test_long_term_memory_performance(self):
        """测试长期记忆性能"""
        # 先创建一些节点
        for i in range(100):
            self.long_term_memory.create_node(f"测试节点{i}")
        
        # 测试长期记忆检索性能
        start_time = time.time()
        nodes = self.long_term_memory.retrieve("测试")
        end_time = time.time()
        retrieve_time = (end_time - start_time) * 1000  # 转换为毫秒
        print(f"长期记忆检索时间: {retrieve_time:.2f} ms")
        self.assertLess(retrieve_time, 100, "长期记忆检索时间应小于100ms")
    
    def test_persona_profile_performance(self):
        """测试 PersonaProfile 记忆痕迹系统性能"""
        # 先存储一些痕迹
        for i in range(100):
            self.persona_profile.store_experience(f"测试经验{i}")
        
        # 测试获取痕迹性能
        trace_id = self.persona_profile.store_experience("性能测试痕迹")
        start_time = time.time()
        trace = self.persona_profile.get_trace(trace_id)
        end_time = time.time()
        get_trace_time = (end_time - start_time) * 1000
        print(f"获取痕迹时间: {get_trace_time:.2f} ms")
        self.assertLess(get_trace_time, 50, "获取痕迹时间应小于50ms")
        
        # 测试条件检索性能
        start_time = time.time()
        traces = self.persona_profile.retrieve_by_conditions(["测试"])
        end_time = time.time()
        retrieve_cond_time = (end_time - start_time) * 1000
        print(f"条件检索时间: {retrieve_cond_time:.2f} ms")
        # 101 个痕迹：SQLite 全表扫描 + 多条件加权，约 150-300ms
        self.assertLess(retrieve_cond_time, 500, "条件检索时间应小于500ms")

if __name__ == '__main__':
    unittest.main()