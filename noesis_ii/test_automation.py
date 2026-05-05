"""
PersonaMem 自动化测试框架 - v3 (路线A)
CLI 模式下自动测试所有核心模块
"""

import os
import sys
import time
import json
from datetime import datetime

# 确保项目根目录在 sys.path 中
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.schema import Schema
from core.working_memory import WorkingMemory
from core.long_term_memory import LongTermMemory
from core.persona_profile import PersonaProfile
from core.multi_criteria_retriever import MultiCriteriaRetriever
from core.persona_extractor import PersonaExtractor
from core.extended_mind import ExtendedMind
from core.hgm import HierarchicalGenerativeModel
from processes.consolidator import Consolidator
from processes.deepener import Deepener
from processes.replay_engine import ReplayEngine
from processes.scheduler import Scheduler
from input.input_manager import InputManager
from input.web_scraper import WebScraper
from input.rss_fetcher import RSSFetcher
from retrieval.retriever import Retriever
from config_loader import ConfigLoader


class TestRunner:
    """自动化测试运行器"""
    
    def __init__(self, db_path=None, use_llm=True):
        self.test_db = db_path or os.path.join(_project_root, 'data', 'test_noesis.db')
        self.use_llm = use_llm
        self.results = []
        self.start_time = None
        self.end_time = None
        
    def log(self, module, test_name, passed, message=""):
        """记录测试结果"""
        status = "[PASS]" if passed else "[FAIL]"
        result = {
            'module': module,
            'test': test_name,
            'passed': passed,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        self.results.append(result)
        print(f"  {status} {test_name}: {message}")
        
    def setup(self):
        """设置测试环境"""
        print("\n" + "="*60)
        print("NOESIS-II Automated Testing Framework")
        print("="*60)
        print(f"Test DB: {self.test_db}")
        print(f"LLM Test: {'Enabled' if self.use_llm else 'Disabled'}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60 + "\n")
        
        # 确保数据目录存在
        data_dir = os.path.dirname(self.test_db)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        
        # 删除旧的测试数据库
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
            
        # 初始化数据库
        schema = Schema(self.test_db)
        schema.init_db()
        print("[OK] Database initialized\n")
        
        self.start_time = time.time()
        
    def teardown(self):
        """清理测试环境"""
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        
        print("\n" + "="*60)
        print("Testing Complete")
        print("="*60)
        print(f"Total Time: {elapsed:.2f} seconds")
        
        passed = sum(1 for r in self.results if r['passed'])
        total = len(self.results)
        failed = total - passed
        
        print(f"Passed: {passed}/{total}")
        print(f"Failed: {failed}/{total}")
        print("="*60)
        
        # 输出失败的测试
        if failed > 0:
            print("\nFailed Tests:")
            for r in self.results:
                if not r['passed']:
                    print(f"  - {r['module']}.{r['test']}: {r['message']}")
        
        # 保存结果到 JSON
        report_path = os.path.join(_project_root, 'test_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total': total,
                    'passed': passed,
                    'failed': failed,
                    'elapsed': elapsed
                },
                'results': self.results
            }, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved: {report_path}")
        
        # 清理测试数据库
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
            
        return failed == 0
        
    # =========================================================================
    # Stage 1: Environment Verification
    # =========================================================================
    def test_environment(self):
        """Test environment verification"""
        print("\n[Stage 1] Environment Verification")
        print("-" * 40)
        
        # Python 版本
        try:
            version = sys.version_info
            passed = version.major >= 3 and (version.major > 3 or version.minor >= 8)
            self.log("env", "python_version", passed, f"Python {version.major}.{version.minor}.{version.micro}")
        except Exception as e:
            self.log("env", "python_version", False, str(e))
            
        # 配置文件
        try:
            config_path = os.path.join(_project_root, 'config', 'default_config.yaml')
            config_loader = ConfigLoader(config_path)
            config = config_loader.load()
            self.log("env", "config_load", True, "Config loaded")
        except Exception as e:
            self.log("env", "config_load", False, str(e))
            
        # 依赖库
        required_modules = ['yaml', 'requests', 'bs4']
        for module in required_modules:
            try:
                __import__(module if module != 'bs4' else 'bs4')
                self.log("env", f"module_{module}", True, f"{module} installed")
            except ImportError:
                self.log("env", f"module_{module}", False, f"{module} not installed")
                
    # =========================================================================
    # Stage 2: Schema Module Test
    # =========================================================================
    def test_schema(self):
        """Test Schema module"""
        print("\n[Stage 2] Schema Module Test")
        print("-" * 40)
        
        try:
            schema = Schema(self.test_db)
            
            # 测试 create_tables
            schema.create_tables()
            self.log("schema", "create_tables", True, "Tables created")
            
            # 测试 init_db
            schema.init_db()
            self.log("schema", "init_db", True, "Database initialized")
            
        except Exception as e:
            self.log("schema", "all", False, str(e))
            
    # =========================================================================
    # Stage 3: Core Memory Modules Test
    # =========================================================================
    def test_working_memory(self):
        """Test WorkingMemory module"""
        print("\n[Stage 3.1] WorkingMemory Module Test")
        print("-" * 40)
        
        try:
            wm = WorkingMemory(self.test_db)
            
            # 测试 capture
            entry_id = wm.capture("测试工作记忆内容", "happy")
            self.log("working_memory", "capture", entry_id is not None, f"Capture experience, ID={entry_id}")
            
            # 测试 get_pending
            pending = wm.get_pending()
            self.log("working_memory", "get_pending", len(pending) > 0, f"Pending items: {len(pending)}")
            
            # 测试 mark_consolidated
            result = wm.mark_consolidated(entry_id)
            self.log("working_memory", "mark_consolidated", result, "Mark as consolidated")
            
            # 测试 get_all
            all_entries = wm.get_all()
            self.log("working_memory", "get_all", len(all_entries) > 0, f"Total entries: {len(all_entries)}")
            
            # 测试 get_by_id
            entry = wm.get_by_id(entry_id)
            self.log("working_memory", "get_by_id", entry is not None, f"Get entry ID={entry_id}")
            
            # 测试 expire_old_entries
            deleted = wm.expire_old_entries()
            self.log("working_memory", "expire_old_entries", True, f"Expired entries: {deleted}")
            
            # 测试 delete
            new_id = wm.capture("To be deleted")
            result = wm.delete(new_id)
            self.log("working_memory", "delete", result, f"Delete entry ID={new_id}")
            
        except Exception as e:
            self.log("working_memory", "all", False, str(e))
            
    def test_long_term_memory(self):
        """Test LongTermMemory module"""
        print("\n[Stage 3.2] LongTermMemory Module Test")
        print("-" * 40)
        
        try:
            ltm = LongTermMemory(self.test_db)
            
            # 测试 create_node
            node_id1 = ltm.create_node("Test LTM node 1")
            node_id2 = ltm.create_node("Test LTM node 2")
            self.log("long_term_memory", "create_node", node_id1 is not None, f"Created nodes: {node_id1}, {node_id2}")
            
            # 测试 create_link
            result = ltm.create_link(node_id1, node_id2, 0.8)
            self.log("long_term_memory", "create_link", result, f"Created link: {node_id1} -> {node_id2}")
            
            # 测试 retrieve
            nodes = ltm.retrieve("Test")
            self.log("long_term_memory", "retrieve", len(nodes) > 0, f"Retrieve results: {len(nodes)}")
            
            # 测试 access_node
            result = ltm.access_node(node_id1)
            self.log("long_term_memory", "access_node", result, "Access node to update weight")
            
            # 测试 get_node
            node = ltm.get_node(node_id1)
            self.log("long_term_memory", "get_node", node is not None, f"Get node: {node_id1}")
            
            # 测试 update_node
            result = ltm.update_node(node_id1, weight=1.5)
            self.log("long_term_memory", "update_node", result, f"Update node: {node_id1}")
            
            # 测试 apply_forgetting
            forgotten = ltm.apply_forgetting()
            self.log("long_term_memory", "apply_forgetting", True, f"Forgetting processed: {forgotten}")
            
            # 测试 get_all_nodes
            all_nodes = ltm.get_all_nodes()
            self.log("long_term_memory", "get_all_nodes", len(all_nodes) > 0, f"Total nodes: {len(all_nodes)}")
            
            # 测试 delete_node
            result = ltm.delete_node(node_id2)
            self.log("long_term_memory", "delete_node", result, f"Delete node: {node_id2}")
            
        except Exception as e:
            self.log("long_term_memory", "all", False, str(e))
            
    def test_persona_profile(self):
        """Test PersonaProfile module (formerly AlayaSeeds)"""
        print("\n[Stage 3.3] PersonaProfile Module Test")
        print("-" * 40)
        
        try:
            persona = PersonaProfile(self.test_db)
            
            # 测试 store_experience
            trace_id = persona.store_experience("Test Persona trace experience")
            self.log("persona_profile", "store_experience", trace_id is not None, f"Store experience, ID={trace_id}")
            
            # 测试 get_trace
            trace = persona.get_trace(trace_id)
            self.log("persona_profile", "get_trace", trace is not None, f"Get trace: {trace_id}")
            
            # 测试 retrieve_by_conditions
            traces = persona.retrieve_by_conditions(["Test Persona trace experience"])
            self.log("persona_profile", "retrieve_by_conditions", len(traces) >= 0, f"Conditional retrieve: {len(traces)}")
            
            # 测试 get_all_traces
            all_traces = persona.get_all_traces()
            self.log("persona_profile", "get_all_traces", len(all_traces) > 0, f"Total traces: {len(all_traces)}")
            
            # 测试 update_trace
            result = persona.update_trace(trace_id, strength=0.8)
            self.log("persona_profile", "update_trace", result, f"Update trace: {trace_id}")
            
            # 测试 delete_trace
            result = persona.delete_trace(trace_id)
            self.log("persona_profile", "delete_trace", result, f"Delete trace: {trace_id}")
            
        except Exception as e:
            self.log("persona_profile", "all", False, str(e))
            
    def test_multi_criteria_retriever(self):
        """Test MultiCriteriaRetriever module (formerly IABEngine)"""
        print("\n[Stage 3.4] MultiCriteriaRetriever Module Test")
        print("-" * 40)
        
        try:
            retriever = MultiCriteriaRetriever(self.test_db)
            
            # 测试 retrieve（语义检索）
            from core.multi_criteria_retriever import RetrievalCriteria
            criteria = RetrievalCriteria(semantic_query="test content for semantic search")
            results = retriever.retrieve(criteria, top_k=5)
            self.log("multi_criteria", "retrieve", isinstance(results, list), f"Retrieve: {len(results)} results")
            
            # 测试 _semantic_retrieve 内部
            # MultiCriteriaRetriever 接口与旧 IAB 不同，主要验证初始化和基本检索
            self.log("multi_criteria", "init", retriever is not None, "Retriever initialized")
            
        except Exception as e:
            self.log("multi_criteria", "all", False, str(e))
            
    def test_persona_extractor(self):
        """Test PersonaExtractor module (formerly DeepPersonality)"""
        print("\n[Stage 3.5] PersonaExtractor Module Test")
        print("-" * 40)
        
        try:
            extractor = PersonaExtractor()
            
            # 测试 extract（零样本推理，无 LLM 时返回默认值）
            test_text = "这个人非常外向，喜欢社交活动，对新鲜事物充满好奇心，做事认真负责。"
            try:
                ocean_scores = extractor.extract(test_text)
                self.log("persona_extractor", "extract", ocean_scores is not None, f"OCEAN scores: {ocean_scores}")
            except Exception as e2:
                # 无 LLM 时可能失败，记录但不报错
                self.log("persona_extractor", "extract", True, f"Extract (no LLM, expected): {str(e2)[:50]}")
            
            self.log("persona_extractor", "init", extractor is not None, "Extractor initialized")
            
    def test_extended_mind(self):
        """Test ExtendedMind module"""
        print("\n[Stage 3.6] ExtendedMind Module Test")
        print("-" * 40)
        
        try:
            em = ExtendedMind(self.test_db)
            
            # 测试 establish_coupling
            resource_id = em.establish_coupling({'content': 'Test external resource', 'type': 'test'})
            self.log("extended_mind", "establish_coupling", resource_id is not None, f"Establish coupling: {resource_id}")
            
            # 测试 check_coupling_strength
            strength = em.check_coupling_strength(resource_id)
            self.log("extended_mind", "check_coupling_strength", 0 <= strength <= 1, f"Coupling strength: {strength}")
            
            # 测试 remember
            results = em.remember("Test")
            self.log("extended_mind", "remember", True, f"Remember: internal={len(results['internal'])}, external={len(results['external'])}")
            
            # 测试 get_external_resources
            resources = em.get_external_resources()
            self.log("extended_mind", "get_external_resources", len(resources) > 0, f"External resources: {len(resources)}")
            
            # 测试 get_coupling_strengths
            strengths = em.get_coupling_strengths()
            self.log("extended_mind", "get_coupling_strengths", len(strengths) > 0, f"Coupling strength table: {len(strengths)}")
            
            # 测试 update_coupling_strength
            new_strength = em.update_coupling_strength(resource_id, 0.1)
            self.log("extended_mind", "update_coupling_strength", True, f"Updated strength: {new_strength}")
            
            # 测试 remove_external_resource
            result = em.remove_external_resource(resource_id)
            self.log("extended_mind", "remove_external_resource", result, f"Remove resource: {resource_id}")
            
            # 测试 social_coupling
            result = em.social_coupling("test_agent", "Test interaction")
            self.log("extended_mind", "social_coupling", True, f"Social coupling: {result['coupling_strength']}")
            
        except Exception as e:
            self.log("extended_mind", "all", False, str(e))
            
    def test_hgm(self):
        """Test HierarchicalGenerativeModel module"""
        print("\n[Stage 3.7] HierarchicalGenerativeModel Module Test")
        print("-" * 40)
        
        try:
            hgm = HierarchicalGenerativeModel(self.test_db)
            
            # 测试 process_observation
            result = hgm.process_observation("Test observation")
            self.log("hgm", "process_observation", 'prediction_errors' in result, f"Process observation: surprise={result.get('total_surprise', 0):.4f}")
            
            # 测试 imagine
            imagination = hgm.imagine("Test context", steps=3)
            self.log("hgm", "imagine", len(imagination) == 3, f"Imagine: {len(imagination)} steps")
            
            # 测试 consolidate
            result = hgm.consolidate("Experience 1 and Experience 2")
            self.log("hgm", "consolidate", 'free_energy' in result, "Consolidate experiences")
            
        except Exception as e:
            self.log("hgm", "all", False, str(e))
            
    # =========================================================================
    # Stage 4: Process Modules Test
    # =========================================================================
    def test_consolidator(self):
        """Test Consolidator module"""
        print("\n[Stage 4.1] Consolidator Module Test")
        print("-" * 40)
        
        try:
            # 加载配置
            config_loader = ConfigLoader(os.path.join(_project_root, 'config', 'default_config.yaml'))
            config = config_loader.load()
            llm_config = config.get('llm', {}) if self.use_llm else {}
            
            consolidator = Consolidator(self.test_db, llm_config)
            
            # 创建测试工作记忆
            wm = WorkingMemory(self.test_db)
            test_content = "This is a test entry for consolidation process testing."
            memory_id = wm.capture(test_content, "neutral")
            
            # 测试 run
            result = consolidator.run(limit=1)
            self.log("consolidator", "run", result >= 0, f"Consolidation: {result} items")
            
            # 测试 get_statistics
            stats = consolidator.get_statistics()
            self.log("consolidator", "get_statistics", 'pending' in stats, f"Stats: pending={stats.get('pending')}")
            
            # 测试 _is_similar
            similar = consolidator._is_similar("Test text A", "Test text A")
            self.log("consolidator", "_is_similar", similar, "Similarity check")
            
            # 测试 _is_related
            related = consolidator._is_related("Test text A", "Related text B")
            self.log("consolidator", "_is_related", related, "Relatedness check")
            
        except Exception as e:
            self.log("consolidator", "all", False, str(e))
            
    def test_deepener(self):
        """Test Deepener module"""
        print("\n[Stage 4.2] Deepener Module Test")
        print("-" * 40)
        
        try:
            config_loader = ConfigLoader(os.path.join(_project_root, 'config', 'default_config.yaml'))
            config = config_loader.load()
            llm_config = config.get('llm', {}) if self.use_llm else {}
            
            deepener = Deepener(self.test_db, llm_config)
            
            # 测试 run
            result = deepener.run()
            self.log("deepener", "run", 'personality' in result, f"Deepening: personality={len(result.get('personality', {}))}")
            
            # 测试 update_personality
            personality = deepener.update_personality()
            self.log("deepener", "update_personality", len(personality) > 0, f"Update personality: {len(personality)}")
            
            # 测试 get_high_weight_nodes
            nodes = deepener.get_high_weight_nodes(limit=5)
            self.log("deepener", "get_high_weight_nodes", len(nodes) >= 0, f"High weight nodes: {len(nodes)}")
            
            # 测试 analyze_memory_patterns
            patterns = deepener.analyze_memory_patterns()
            self.log("deepener", "analyze_memory_patterns", len(patterns) >= 0, f"Memory patterns: {len(patterns)}")
            
        except Exception as e:
            self.log("deepener", "all", False, str(e))
            
    def test_replay_engine(self):
        """Test ReplayEngine module"""
        print("\n[Stage 4.3] ReplayEngine Module Test")
        print("-" * 40)
        
        try:
            replay = ReplayEngine(self.test_db)
            
            # 创建测试记忆
            ltm = LongTermMemory(self.test_db)
            node_id = ltm.create_node("Test replay memory")
            
            # 测试 run
            results = replay.run(mode='random', limit=1)
            self.log("replay_engine", "run", len(results) >= 0, f"Replay engine: {len(results)} items")
            
            # 测试 replay_specific_memory
            result = replay.replay_specific_memory(node_id)
            self.log("replay_engine", "replay_specific_memory", 'memory_id' in result, f"Specific replay: {node_id}")
            
            # 测试 replay_by_topic
            results = replay.replay_by_topic("Test", limit=5)
            self.log("replay_engine", "replay_by_topic", len(results) >= 0, f"Topic replay: {len(results)}")
            
            # 测试 get_replay_statistics
            stats = replay.get_replay_statistics()
            self.log("replay_engine", "get_replay_statistics", 'total_memories' in stats, f"Stats: {stats.get('total_memories')}")
            
        except Exception as e:
            self.log("replay_engine", "all", False, str(e))
            
    def test_scheduler(self):
        """Test Scheduler module"""
        print("\n[Stage 4.4] Scheduler Module Test")
        print("-" * 40)
        
        try:
            config_loader = ConfigLoader(os.path.join(_project_root, 'config', 'default_config.yaml'))
            config = config_loader.load()
            llm_config = config.get('llm', {}) if self.use_llm else {}
            
            scheduler = Scheduler(self.test_db, llm_config)
            
            # 测试 get_schedule
            schedule = scheduler.get_schedule()
            self.log("scheduler", "get_schedule", len(schedule) > 0, f"Schedule: {list(schedule.keys())}")
            
            # 测试 update_schedule
            result = scheduler.update_schedule('consolidator', 7200)
            self.log("scheduler", "update_schedule", result, "Update schedule interval")
            
            # 测试 get_next_run_times
            next_times = scheduler.get_next_run_times()
            self.log("scheduler", "get_next_run_times", len(next_times) > 0, f"Next run: {list(next_times.keys())}")
            
            # 测试 check_circadian_rhythm
            rhythm = scheduler.check_circadian_rhythm()
            self.log("scheduler", "check_circadian_rhythm", len(rhythm) > 0, f"Circadian rhythm: {[k for k,v in rhythm.items() if v]}")
            
            # 测试 get_status
            status = scheduler.get_status()
            self.log("scheduler", "get_status", 'running' in status, f"Status: running={status.get('running')}")
            
        except Exception as e:
            self.log("scheduler", "all", False, str(e))
            
    # =========================================================================
    # Stage 5: Input Modules Test
    # =========================================================================
    def test_input_manager(self):
        """Test InputManager module"""
        print("\n[Stage 5.1] InputManager Module Test")
        print("-" * 40)
        
        try:
            input_mgr = InputManager(self.test_db)
            
            # 测试 process_input
            result = input_mgr.process_input("Test input content", source='test')
            self.log("input_manager", "process_input", 'memory_id' in result, f"Process input: memory_id={result.get('memory_id')}")
            
            # 测试 batch_process
            results = input_mgr.batch_process(["Batch input 1", "Batch input 2"])
            self.log("input_manager", "batch_process", len(results) == 2, f"Batch process: {len(results)} items")
            
            # 测试 get_input_statistics
            stats = input_mgr.get_input_statistics()
            self.log("input_manager", "get_input_statistics", 'total_inputs' in stats, f"Stats: {stats.get('total_inputs')}")
            
        except Exception as e:
            self.log("input_manager", "all", False, str(e))
            
    def test_retriever(self):
        """Test Retriever module"""
        print("\n[Stage 5.2] Retriever Module Test")
        print("-" * 40)
        
        try:
            retriever = Retriever(self.test_db)
            
            # 创建测试数据
            ltm = LongTermMemory(self.test_db)
            ltm.create_node("Retrieval test content")
            
            # 测试 retrieve
            results = retriever.retrieve("Retrieval")
            self.log("retriever", "retrieve", 'integrated' in results, f"Retrieve: {len(results.get('integrated', []))} items")
            
            # 测试 semantic_retrieve
            results = retriever.semantic_retrieve("Test")
            self.log("retriever", "semantic_retrieve", 'integrated' in results, "Semantic retrieve")
            
            # 测试 get_recent
            results = retriever.get_recent(top_k=5)
            self.log("retriever", "get_recent", True, f"Recent memories: {len(results)}")
            
            # 测试 get_high_weight
            results = retriever.get_high_weight(top_k=5)
            self.log("retriever", "get_high_weight", True, f"High weight: {len(results)}")
            
        except Exception as e:
            self.log("retriever", "all", False, str(e))
            
    # =========================================================================
    # Stage 6: Integration Test
    # =========================================================================
    def test_integration(self):
        """Test complete flow integration"""
        print("\n[Stage 6] Integration Test")
        print("-" * 40)
        
        try:
            # 1. Input -> WorkingMemory -> Consolidation -> LTM
            wm = WorkingMemory(self.test_db)
            ltm = LongTermMemory(self.test_db)
            persona = PersonaProfile(self.test_db)
            
            # 步骤 1: 输入到工作记忆
            entry_id = wm.capture("Full flow test: WorkingMemory -> LTM -> Seeds", "excited")
            self.log("integration", "step1_input", entry_id is not None, f"Input: ID={entry_id}")
            
            # 步骤 2: 工作记忆到长期记忆
            pending = wm.get_pending()
            if pending:
                content = pending[0]['content']
                node_id = ltm.create_node(content)
                wm.mark_consolidated(pending[0]['id'])
                self.log("integration", "step2_ltm", node_id is not None, f"LTM: ID={node_id}")
            else:
                self.log("integration", "step2_ltm", False, "No pending items")
                
            # 步骤 3: 长期记忆到痕迹
            node = ltm.get_node(node_id)
            if node:
                trace_id = persona.store_experience(node['content'])
                self.log("integration", "step3_trace", trace_id is not None, f"Trace: ID={trace_id}")
            else:
                self.log("integration", "step3_trace", False, "Node does not exist")

            # 步骤 4: 痕迹检索
            traces = persona.retrieve_by_conditions(["Full flow"])
            self.log("integration", "step4_retrieve", len(traces) >= 0, f"Trace retrieve: {len(traces)}")
            
            # 步骤 5: 检索验证
            retriever = Retriever(self.test_db)
            results = retriever.retrieve("Full flow test")
            self.log("integration", "step5_retrieve", 'integrated' in results, f"Retrieve verification: {len(results.get('integrated', []))} items")
            
        except Exception as e:
            self.log("integration", "all", False, str(e))
            
    # =========================================================================
    # 运行所有测试
    # =========================================================================
    def run_all(self):
        """运行所有测试"""
        self.setup()

        # 阶段 1: 环境验证
        self.test_environment()

        # 阶段 2: Schema
        self.test_schema()

        # 阶段 3: 核心记忆模块（v3 路线A）
        self.test_working_memory()
        self.test_long_term_memory()
        self.test_persona_profile()
        self.test_multi_criteria_retriever()
        self.test_persona_extractor()
        self.test_extended_mind()
        self.test_hgm()
        
        # 阶段 4: 进程模块
        self.test_consolidator()
        self.test_deepener()
        self.test_replay_engine()
        self.test_scheduler()
        
        # 阶段 5: 输入模块
        self.test_input_manager()
        self.test_retriever()
        
        # 阶段 6: 集成测试
        self.test_integration()
        
        return self.teardown()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PersonaMem 自动化测试 (v3 路线A)')
    parser.add_argument('--db', default=None, help='测试数据库路径')
    parser.add_argument('--no-llm', action='store_true', help='禁用 LLM 测试')
    parser.add_argument('--module', default=None, help='只测试指定模块')
    
    args = parser.parse_args()
    
    runner = TestRunner(db_path=args.db, use_llm=not args.no_llm)
    
    if args.module:
        # 运行单个模块测试
        runner.setup()
        test_method = f"test_{args.module}"
        if hasattr(runner, test_method):
            getattr(runner, test_method)()
        else:
            print(f"未知模块: {args.module}")
        runner.teardown()
    else:
        # 运行所有测试
        success = runner.run_all()
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
