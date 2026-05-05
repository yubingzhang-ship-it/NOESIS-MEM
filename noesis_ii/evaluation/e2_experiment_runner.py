"""
E2 一致性实验运行器

执行跨 Session 的一致性测试
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import os

from .e2_question_bank import QuestionBank, Question
from .e2_response_extractor import (
    ResponseExtractor, 
    ExtractedResponse,
    compute_ocean_stability,
    compute_ocean_cosine
)
from .e2_group_manager import (
    GroupManagerFactory,
    BaseGroupManager,
    SessionResult
)


@dataclass
class ExperimentConfig:
    """实验配置"""
    session_gap_days: int = 7  # Session A 和 B 的间隔天数
    use_shuffled_questions: bool = True  # Session B 是否打乱题目顺序
    save_intermediate: bool = True  # 保存中间结果
    output_dir: str = "evaluation_results/e2"
    
    # 评估阈值（研究计划定义）
    stability_threshold: float = 0.10  # OCEAN σ < 0.10
    cosine_threshold: float = 0.85  # Cosine > 0.85
    conflict_threshold: float = 0.05  # 冲突率 < 5%


@dataclass
class ExperimentResult:
    """实验结果"""
    group: str
    session_a: SessionResult
    session_b: SessionResult
    
    # 一致性指标
    ocean_stability: Dict[str, float]  # 各维度 σ
    ocean_cosine: float  # 向量余弦相似度
    conflict_rate: float  # 冲突率
    
    # 判断
    passed: bool
    passed_dimensions: List[str]
    failed_dimensions: List[str]
    
    # 详细分析
    detailed_analysis: Dict[str, Any] = field(default_factory=dict)


class E2ExperimentRunner:
    """
    E2 一致性实验运行器
    
    执行流程：
    1. 初始化三个实验组
    2. 运行 Session A（收集初始响应）
    3. 模拟时间间隔（Ours 组会积累记忆）
    4. 运行 Session B（收集响应）
    5. 计算一致性指标
    6. 生成报告
    """
    
    def __init__(
        self,
        config: ExperimentConfig = None,
        llm_client = None
    ):
        self.config = config or ExperimentConfig()
        self.llm = llm_client
        
        # 初始化组件
        self.question_bank = QuestionBank()
        self.response_extractor = ResponseExtractor(llm_client)
        
        # 初始化实验组
        self.groups: Dict[str, BaseGroupManager] = {}
        
        # 实验结果
        self.results: Dict[str, ExperimentResult] = {}
    
    def initialize_groups(
        self,
        bl2_persona: str = None,
        ours_profile_path: str = None
    ):
        """初始化实验组"""
        self.groups = GroupManagerFactory.create_all(
            llm_client=self.llm,
            bl2_persona=bl2_persona,
            ours_profile_path=ours_profile_path
        )
        
        # 确保输出目录存在
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    async def run_session_a(self) -> Dict[str, SessionResult]:
        """
        运行 Session A（第一天）
        
        Returns:
            各组的 Session A 结果
        """
        print("=" * 60)
        print("Session A 开始")
        print("=" * 60)
        
        questions = self.question_bank.get_all()
        session_results = {}
        timestamp = datetime.now().isoformat()
        
        for group_name, manager in self.groups.items():
            print(f"\n[{group_name}] 回答 {len(questions)} 道题...")
            
            responses = []
            ocean_profiles = []
            
            for i, question in enumerate(questions):
                # 构建问题
                question_text = self.question_bank.format_for_llm(question)
                messages = [{"role": "user", "content": question_text}]
                
                # 获取回答
                try:
                    response = await manager.chat(messages)
                except Exception as e:
                    print(f"  [警告] 第 {i+1} 题回答失败: {e}")
                    response = "A"  # fallback
                
                # 提取回答
                extracted = self.response_extractor.extract(response, question)
                
                responses.append({
                    'question_id': question.id,
                    'question_text': question.text,
                    'choice': extracted.choice,
                    'ocean_scores': extracted.ocean_scores,
                    'reasoning': extracted.reasoning,
                    'confidence': extracted.confidence
                })
                
                ocean_profiles.append(extracted.ocean_scores)
                
                if (i + 1) % 5 == 0:
                    print(f"  进度: {i+1}/{len(questions)}")
            
            # 创建 Session 结果
            session_result = SessionResult(
                session_id=f"{group_name}_A",
                group=group_name,
                timestamp=timestamp,
                responses=responses,
                ocean_profile=self._aggregate_ocean(ocean_profiles)
            )
            
            manager.record_session(session_result)
            session_results[group_name] = session_result
            
            # 保存中间结果
            if self.config.save_intermediate:
                self._save_session_result(session_result)
            
            print(f"[{group_name}] Session A 完成 ✓")
        
        return session_results
    
    async def run_session_b(
        self,
        session_a_results: Dict[str, SessionResult]
    ) -> Dict[str, SessionResult]:
        """
        运行 Session B（模拟一周后）
        
        对于 Ours 组，期间会进行一些"日常对话"来积累记忆
        """
        print("\n" + "=" * 60)
        print("Session B 开始 (模拟一周后)")
        print("=" * 60)
        
        # 获取题目（可选打乱）
        questions = (self.question_bank.get_shuffled() 
                    if self.config.use_shuffled_questions 
                    else self.question_bank.get_all())
        
        session_results = {}
        timestamp = datetime.now().isoformat()
        
        for group_name, manager in self.groups.items():
            print(f"\n[{group_name}] 回答 {len(questions)} 道题...")
            
            # 对于 Ours 组，模拟一周的日常对话
            if group_name == 'Ours':
                await self._simulate_daily_conversations(manager)
            
            responses = []
            ocean_profiles = []
            
            for i, question in enumerate(questions):
                question_text = self.question_bank.format_for_llm(question)
                messages = [{"role": "user", "content": question_text}]
                
                try:
                    response = await manager.chat(messages)
                except Exception as e:
                    print(f"  [警告] 第 {i+1} 题回答失败: {e}")
                    response = "B"  # fallback
                
                extracted = self.response_extractor.extract(response, question)
                
                responses.append({
                    'question_id': question.id,
                    'question_text': question.text,
                    'choice': extracted.choice,
                    'ocean_scores': extracted.ocean_scores,
                    'reasoning': extracted.reasoning,
                    'confidence': extracted.confidence
                })
                
                ocean_profiles.append(extracted.ocean_scores)
                
                if (i + 1) % 5 == 0:
                    print(f"  进度: {i+1}/{len(questions)}")
            
            session_result = SessionResult(
                session_id=f"{group_name}_B",
                group=group_name,
                timestamp=timestamp,
                responses=responses,
                ocean_profile=self._aggregate_ocean(ocean_profiles)
            )
            
            manager.record_session(session_result)
            session_results[group_name] = session_result
            
            if self.config.save_intermediate:
                self._save_session_result(session_result)
            
            print(f"[{group_name}] Session B 完成 ✓")
        
        return session_results
    
    async def _simulate_daily_conversations(self, manager):
        """
        模拟一周的日常对话
        
        用于 Ours 组在 Session A 和 B 之间积累记忆
        """
        print("  [Ours] 模拟一周日常对话...")
        
        daily_topics = [
            "今天天气不错，心情如何？",
            "周末有什么计划吗？",
            "最近在工作或学习上有什么进展？",
            "有没有什么想尝试的新事物？",
            "和朋友相处时你更看重什么？",
            "面对压力时你通常怎么应对？",
        ]
        
        for topic in daily_topics:
            messages = [{"role": "user", "content": topic}]
            try:
                await manager.chat(messages)
            except Exception:
                pass  # 忽略日常对话的错误
        
        print(f"  [Ours] 完成 {len(daily_topics)} 次日常对话")
    
    def calculate_metrics(
        self,
        session_a: SessionResult,
        session_b: SessionResult
    ) -> ExperimentResult:
        """
        计算一致性指标
        """
        # 计算 OCEAN 稳定性
        ocean_a = [r['ocean_scores'] for r in session_a.responses]
        ocean_b = [r['ocean_scores'] for r in session_b.responses]
        
        stability = compute_ocean_stability(ocean_a, ocean_b)
        cosine = compute_ocean_cosine(ocean_a, ocean_b)
        
        # 计算冲突率（同题不同选项的比例）
        conflicts = 0
        total = len(session_a.responses)
        
        for resp_a, resp_b in zip(session_a.responses, session_b.responses):
            if resp_a['choice'] != resp_b['choice']:
                conflicts += 1
        
        conflict_rate = conflicts / total if total > 0 else 0
        
        # 判断各维度是否达标
        dimensions = ['O', 'C', 'E', 'A', 'N']
        passed_dims = [d for d in dimensions if stability[d] < self.config.stability_threshold]
        failed_dims = [d for d in dimensions if d not in passed_dims]
        
        # 综合判断
        passed = (
            all(stability[d] < self.config.stability_threshold for d in dimensions) and
            cosine > self.config.cosine_threshold and
            conflict_rate < self.config.conflict_threshold
        )
        
        return ExperimentResult(
            group=session_a.group,
            session_a=session_a,
            session_b=session_b,
            ocean_stability=stability,
            ocean_cosine=cosine,
            conflict_rate=conflict_rate,
            passed=passed,
            passed_dimensions=passed_dims,
            failed_dimensions=failed_dims,
            detailed_analysis={
                'total_questions': total,
                'choice_conflicts': conflicts,
                'stability_threshold': self.config.stability_threshold,
                'cosine_threshold': self.config.cosine_threshold,
                'conflict_threshold': self.config.conflict_threshold
            }
        )
    
    def run_full_experiment(self) -> Dict[str, ExperimentResult]:
        """
        运行完整实验
        """
        # 检查是否已初始化
        if not self.groups:
            self.initialize_groups()
        
        # 运行 Session A
        session_a_results = asyncio.run(self.run_session_a())
        
        # 运行 Session B
        session_b_results = asyncio.run(self.run_session_b(session_a_results))
        
        # 计算各组指标
        for group_name in self.groups:
            result = self.calculate_metrics(
                session_a_results[group_name],
                session_b_results[group_name]
            )
            self.results[group_name] = result
        
        # 生成报告
        self._generate_report()
        
        return self.results
    
    def _aggregate_ocean(self, profiles: List[Dict[str, float]]) -> Dict[str, float]:
        """聚合 OCEAN 画像"""
        dimensions = ['O', 'C', 'E', 'A', 'N']
        aggregated = {}
        
        for dim in dimensions:
            scores = [p[dim] for p in profiles]
            aggregated[dim] = sum(scores) / len(scores) if scores else 0.5
        
        return aggregated
    
    def _save_session_result(self, session_result: SessionResult):
        """保存 session 结果"""
        filename = f"{self.config.output_dir}/{session_result.session_id}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'session_id': session_result.session_id,
                'group': session_result.group,
                'timestamp': session_result.timestamp,
                'responses': session_result.responses,
                'ocean_profile': session_result.ocean_profile,
                'metadata': session_result.metadata
            }, f, ensure_ascii=False, indent=2)
    
    def _generate_report(self):
        """生成实验报告"""
        report_path = f"{self.config.output_dir}/e2_experiment_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# E2 跨Session一致性实验报告\n\n")
            f.write(f"**实验时间**: {datetime.now().isoformat()}\n\n")
            
            # 各组结果
            for group, result in self.results.items():
                f.write(f"## {group}\n\n")
                
                status = "✅ **通过**" if result.passed else "❌ **未通过**"
                f.write(f"**状态**: {status}\n\n")
                
                # 一致性指标
                f.write("### 一致性指标\n\n")
                f.write("| 维度 | 稳定性 (σ) | 阈值 | 状态 |\n")
                f.write("|------|-----------|------|------|\n")
                
                for dim in ['O', 'C', 'E', 'A', 'N']:
                    sigma = result.ocean_stability[dim]
                    threshold = self.config.stability_threshold
                    status_icon = "✅" if sigma < threshold else "❌"
                    f.write(f"| {dim} | {sigma:.4f} | {threshold} | {status_icon} |\n")
                
                f.write(f"\n| 指标 | 实际值 | 阈值 | 状态 |\n")
                f.write(f"|------|-------|------|------|\n")
                f.write(f"| Cosine | {result.ocean_cosine:.4f} | >{self.config.cosine_threshold} | ")
                f.write("✅" if result.ocean_cosine > self.config.cosine_threshold else "❌")
                f.write(" |\n")
                f.write(f"| 冲突率 | {result.conflict_rate:.2%} | <{self.config.conflict_threshold:.2%} | ")
                f.write("✅" if result.conflict_rate < self.config.conflict_threshold else "❌")
                f.write(" |\n")
                
                f.write("\n### 达标情况\n\n")
                f.write(f"- 通过维度: {', '.join(result.passed_dimensions) if result.passed_dimensions else '无'}\n")
                f.write(f"- 未通过维度: {', '.join(result.failed_dimensions) if result.failed_dimensions else '无'}\n")
                f.write("\n---\n\n")
            
            # 汇总对比
            f.write("## 汇总对比\n\n")
            f.write("| 组别 | 通过 | O σ均值 | Cosine | 冲突率 |\n")
            f.write("|------|------|---------|--------|--------|\n")
            
            for group, result in self.results.items():
                sigma_mean = sum(result.ocean_stability.values()) / 5
                status = "✅" if result.passed else "❌"
                f.write(f"| {group} | {status} | {sigma_mean:.4f} | {result.ocean_cosine:.4f} | {result.conflict_rate:.2%} |\n")
        
        print(f"\n报告已保存: {report_path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取结果摘要"""
        summary = {
            'groups': {},
            'overall_passed': all(r.passed for r in self.results.values()),
            'best_group': None,
            'config': {
                'session_gap_days': self.config.session_gap_days,
                'use_shuffled_questions': self.config.use_shuffled_questions
            }
        }
        
        # 找出最佳组
        if self.results:
            best = max(self.results.items(), key=lambda x: (
                x[1].ocean_cosine,
                -x[1].conflict_rate,
                -sum(x[1].ocean_stability.values()) / 5
            ))
            summary['best_group'] = best[0]
        
        for group, result in self.results.items():
            summary['groups'][group] = {
                'passed': result.passed,
                'cosine': result.ocean_cosine,
                'conflict_rate': result.conflict_rate,
                'stability_mean': sum(result.ocean_stability.values()) / 5,
                'passed_dims': result.passed_dimensions
            }
        
        return summary


async def quick_demo():
    """快速演示（不调用 LLM）"""
    from .e2_group_manager import BL1GroupManager, BL2GroupManager, OursGroupManager
    
    runner = E2ExperimentRunner()
    runner.groups = {
        'BL-1': BL1GroupManager(),
        'BL-2': BL2GroupManager(),
        'Ours': OursGroupManager()
    }
    
    # 运行模拟
    questions = runner.question_bank.get_all()
    
    for group_name, manager in runner.groups.items():
        print(f"\n[{group_name}] Demo 运行...")
        
        # 模拟回答
        for q in questions[:3]:  # 只跑3题演示
            print(f"  Q{q.id}: A")
    
    print("\n演示完成！")


if __name__ == "__main__":
    asyncio.run(quick_demo())
