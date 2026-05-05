import sqlite3
import os
import datetime
from core.working_memory import WorkingMemory
from core.multi_criteria_retriever import MultiCriteriaRetriever

class InputManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.working_memory = WorkingMemory(db_path)
        self.retriever = MultiCriteriaRetriever(db_path)
        self.input_sources = []
        self.active_sources = set()
    
    def register_input_source(self, source_name, source):
        """Register an input source"""
        self.input_sources.append({'name': source_name, 'source': source})
        print(f"[INPUT] Registered input source: {source_name}")

    def activate_source(self, source_name):
        """Activate an input source"""
        for source in self.input_sources:
            if source['name'] == source_name:
                self.active_sources.add(source_name)
                print(f"[INPUT] Activated: {source_name}")
                return True
        print(f"[INPUT] Source not found: {source_name}")
        return False

    def deactivate_source(self, source_name):
        """Deactivate an input source"""
        if source_name in self.active_sources:
            self.active_sources.remove(source_name)
            print(f"[INPUT] Deactivated: {source_name}")
            return True
        print(f"[INPUT] Source not active: {source_name}")
        return False
    
    def process_input(self, input_data, input_type='text', source='user'):
        """处理输入"""
        # 预处理输入
        processed_input = self._preprocess_input(input_data, input_type)
        
        # 存储到工作记忆
        memory_id = self.working_memory.capture(processed_input, emotion=None)

        # 检索相关记忆（替代原意识处理）
        from .multi_criteria_retriever import RetrievalCriteria
        criteria = RetrievalCriteria(semantic_query=processed_input[:200] if len(processed_input) > 200 else processed_input)
        related_memories = self.retriever.retrieve(criteria, top_k=5)
        
        return {
            'input': input_data,
            'processed_input': processed_input,
            'memory_id': memory_id,
            'related_count': len(related_memories)
        }
    
    def _preprocess_input(self, input_data, input_type):
        """预处理输入数据"""
        if input_type == 'text':
            # 文本预处理
            return self._preprocess_text(input_data)
        elif input_type == 'voice':
            # 语音预处理
            return self._preprocess_voice(input_data)
        elif input_type == 'image':
            # 图像预处理
            return self._preprocess_image(input_data)
        else:
            # 默认处理
            return str(input_data)
    
    def _preprocess_text(self, text):
        """预处理文本输入"""
        # 简化实现，实际项目中应该有更复杂的文本预处理
        return text.strip()
    
    def _preprocess_voice(self, voice_data):
        """Preprocess voice input"""
        return f"[VOICE] {voice_data}"

    def _preprocess_image(self, image_data):
        """Preprocess image input"""
        return f"[IMAGE] {image_data}"
    
    def get_active_sources(self):
        """获取活跃输入源"""
        return list(self.active_sources)
    
    def get_all_sources(self):
        """获取所有输入源"""
        return [source['name'] for source in self.input_sources]
    
    def run_active_sources(self):
        """运行活跃输入源"""
        results = {}
        for source in self.input_sources:
            if source['name'] in self.active_sources:
                try:
                    # 运行输入源
                    result = source['source'].run()
                    results[source['name']] = result
                    
                    # 处理输入结果
                    if result and 'content' in result:
                        self.process_input(result['content'], source=source['name'])
                except Exception as e:
                    results[source['name']] = {'error': str(e)}
                    print(f"[INPUT] Source '{source['name']}' failed: {e}")
        return results
    
    def schedule_input(self, input_data, delay=0):
        """调度输入"""
        # 简化实现，实际项目中应该使用调度器
        if delay > 0:
            import time
            time.sleep(delay)
        return self.process_input(input_data)
    
    def batch_process(self, inputs):
        """批量处理输入"""
        results = []
        for input_item in inputs:
            if isinstance(input_item, dict):
                result = self.process_input(
                    input_item.get('data'),
                    input_item.get('type', 'text'),
                    input_item.get('source', 'user')
                )
            else:
                result = self.process_input(input_item)
            results.append(result)
        return results
    
    def get_input_statistics(self):
        """获取输入统计信息"""
        # 获取工作记忆中的输入数量
        all_entries = self.working_memory.get_all()
        total_inputs = len(all_entries)
        
        # 获取关联记忆数量
        from .multi_criteria_retriever import RetrievalCriteria
        criteria = RetrievalCriteria()
        related_memories = self.retriever.retrieve(criteria, top_k=1)
        related_count = len(related_memories)
        
        return {
            'total_inputs': total_inputs,
            'related_count': related_count,
            'active_sources': len(self.active_sources),
            'total_sources': len(self.input_sources)
        }