"""
ConsistencyChecker - 人格一致性检查器

Week 5-6 任务：一致性检查器增强

核心功能：
- 检查生成内容是否符合历史人格分布
- 计算生成内容与目标人格的 KL 散度
- 价值观冲突检测（基于 LLM）
- 提供重新生成决策建议

修订历史：
  v1.0 (2026-04-10) - 路线A新增模块
  v1.1 (2026-04-10) - Week 5-6: 阈值调整 0.5→0.70，集成价值观冲突检测
"""

import json
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ConsistencyReport:
    """一致性检查报告"""
    is_consistent: bool           # 是否一致
    kl_divergence: float          # KL散度
    drift_score: float            # 漂移分数 [0, 1]
    dimension_drifts: Dict[str, float]  # 各维度漂移
    value_conflict: bool = False  # 价值观冲突
    conflict_reason: Optional[str] = None  # 冲突原因
    suggestions: List[str] = field(default_factory=list)  # 改进建议
    should_regenerate: bool = False  # 是否应该重新生成


class ConsistencyChecker:
    """
    人格一致性检查器
    
    检查生成内容是否符合历史人格分布，
    包括 OCEAN 维度一致性和价值观冲突检测。
    
    Week 5-6 增强：
    - 阈值从 0.5 调整为 0.70（研究计划推荐）
    - 集成 ValueConflictDetector
    - 决策：是否触发重新生成
    """
    
    def __init__(self, llm_client=None, value_detector=None):
        self.llm = llm_client
        self.value_detector = value_detector
        
        # Week 5-6: 阈值调整到 0.70（研究计划推荐）
        self.kl_threshold = 0.70
        
        # 漂移警告阈值
        self.drift_warning = 0.3
        self.drift_critical = 0.5
        
        # 重新生成决策阈值
        # 满足以下任一条件时触发重新生成：
        # 1. KL 散度 > kl_threshold
        # 2. 价值观冲突
        # 3. 任意维度漂移 > drift_critical
    
    def check_consistency(
        self,
        generated_text: str,
        target_persona: Dict[str, float],
        extractor=None,
        persona_history: List[Dict] = None
    ) -> ConsistencyReport:
        """
        检查生成内容的人格一致性
        
        Week 5-6 增强：
        - 集成价值观冲突检测
        - 返回 should_regenerate 决策
        
        Args:
            generated_text: 生成的文本
            target_persona: 目标人格分布（OCEAN分数）
            extractor: 人格提取器（用于提取生成内容的人格）
            persona_history: 人格历史记录（用于价值观冲突检测）
            
        Returns:
            ConsistencyReport: 一致性检查报告
        """
        if extractor is None:
            return self._fallback_check(generated_text, target_persona)
        
        # 提取生成内容的人格
        from .persona_extractor import PersonaExtractor
        if isinstance(extractor, PersonaExtractor):
            actual_persona = extractor.extract_ocean(generated_text).to_dict()
        else:
            actual_persona = extractor(generated_text)
        
        # 计算 KL 散度
        kl_div = self._compute_kl_divergence(target_persona, actual_persona)
        
        # 计算各维度漂移
        dimension_drifts = {}
        for dim in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
            target = target_persona.get(dim, 0.5)
            actual = actual_persona.get(dim, 0.5)
            dimension_drifts[dim] = abs(target - actual)
        
        # 计算总体漂移分数
        drift_score = sum(dimension_drifts.values()) / len(dimension_drifts)
        
        # Week 5-6: 价值观冲突检测
        value_conflict = False
        conflict_reason = None
        if self.value_detector and persona_history:
            value_conflict, conflict_reason = self.value_detector.detect_conflict(
                generated_text, persona_history
            )
        
        # 判断是否一致（Week 5-6: 综合判断）
        is_consistent = (
            kl_div < self.kl_threshold and 
            drift_score < self.drift_warning and 
            not value_conflict
        )
        
        # 判断是否需要重新生成
        should_regenerate = (
            kl_div > self.kl_threshold or 
            value_conflict or
            any(d > self.drift_critical for d in dimension_drifts.values())
        )
        
        # 生成建议
        suggestions = self._generate_suggestions(
            dimension_drifts, kl_div, value_conflict, conflict_reason
        )
        
        return ConsistencyReport(
            is_consistent=is_consistent,
            kl_divergence=kl_div,
            drift_score=drift_score,
            dimension_drifts=dimension_drifts,
            value_conflict=value_conflict,
            conflict_reason=conflict_reason,
            suggestions=suggestions,
            should_regenerate=should_regenerate
        )
    
    def _compute_kl_divergence(
        self,
        target: Dict[str, float],
        actual: Dict[str, float]
    ) -> float:
        """
        计算人格偏差距离（欧氏距离）
        
        OCEAN分数不是概率分布，KL散度在此场景下不适用。
        使用欧氏距离作为替代度量。
        """
        total_sq_diff = 0.0
        
        for dim in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
            p = target.get(dim, 0.5)
            q = actual.get(dim, 0.5)
            total_sq_diff += (p - q) ** 2
        
        euclidean = math.sqrt(total_sq_diff)
        # 归一化到 [0, 1]
        return euclidean / math.sqrt(5)
    
    def _generate_suggestions(
        self,
        dimension_drifts: Dict[str, float],
        kl_div: float,
        value_conflict: bool = False,
        conflict_reason: Optional[str] = None
    ) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        # 找出漂移最大的维度
        max_drift_dim = max(dimension_drifts.items(), key=lambda x: x[1])
        
        # KL 散度建议
        if kl_div > self.kl_threshold:
            suggestions.append(
                f"[人格偏离] KL散度 {kl_div:.2f} > 阈值 {self.kl_threshold:.2f}，"
                f"建议重新生成"
            )
        
        # 价值观冲突建议
        if value_conflict:
            reason = conflict_reason or "检测到价值观矛盾"
            suggestions.append(f"[价值观冲突] {reason}，强烈建议重新生成")
        
        # 维度漂移建议
        if max_drift_dim[1] > self.drift_critical:
            dim_name = self._translate_dimension(max_drift_dim[0])
            suggestions.append(
                f"[维度漂移] {dim_name}维度漂移严重 ({max_drift_dim[1]:.2f})，"
                f"需在重新生成时强调该维度"
            )
        elif max_drift_dim[1] > self.drift_warning:
            dim_name = self._translate_dimension(max_drift_dim[0])
            suggestions.append(
                f"[轻微漂移] {dim_name}维度有轻微漂移 ({max_drift_dim[1]:.2f})，"
                f"可接受但建议监控"
            )
        
        # 如果没有问题
        if not suggestions:
            suggestions.append("[OK] 人格一致性检查通过")
        
        return suggestions
    
    def _translate_dimension(self, dim: str) -> str:
        """翻译维度名称"""
        translations = {
            'openness': '开放性',
            'conscientiousness': '尽责性',
            'extraversion': '外向性',
            'agreeableness': '宜人性',
            'neuroticism': '神经质'
        }
        return translations.get(dim, dim)
    
    def _fallback_check(
        self,
        text: str,
        target_persona: Dict[str, float]
    ) -> ConsistencyReport:
        """备用检查方案"""
        # 简化的启发式检查
        return ConsistencyReport(
            is_consistent=True,
            kl_divergence=0.0,
            drift_score=0.0,
            dimension_drifts={},
            value_conflict=False,
            conflict_reason=None,
            suggestions=["[INFO] 未配置人格提取器，跳过一致性检查"],
            should_regenerate=False
        )
    
    def compute_decoding_penalty(
        self,
        token_probs: Dict[str, float],
        target_persona: Dict[str, float]
    ) -> Dict[str, float]:
        """
        计算解码时的 token 惩罚（未来扩展）
        
        目标：对可能导致人格偏离的 token 施加惩罚
        
        Args:
            token_probs: token 概率分布
            target_persona: 目标人格
            
        Returns:
            每个 token 的惩罚值
        """
        # TODO: 实现基于人格约束的解码惩罚
        # 这是未来工作，需要与 LLM 解码过程深度集成
        
        return {token: 0.0 for token in token_probs}
    
    def get_regeneration_advice(
        self,
        report: ConsistencyReport,
        current_persona: Dict[str, float]
    ) -> str:
        """
        获取重新生成的建议提示（用于传给 LLM）
        
        Args:
            report: 一致性检查报告
            current_persona: 当前人格
            
        Returns:
            用于重新生成的增强提示
        """
        if not report.should_regenerate:
            return ""
        
        parts = ["[人格一致性检查未通过，请调整生成策略]"]
        
        # KL 散度
        if report.kl_divergence > self.kl_threshold:
            parts.append(
                f"\n1. KL散度过高 ({report.kl_divergence:.2f})，"
                f"当前人格分布与目标偏离较大"
            )
        
        # 价值观冲突
        if report.value_conflict:
            parts.append(f"\n2. 价值观冲突: {report.conflict_reason or '检测到矛盾'}")
        
        # 维度漂移
        critical_dims = [
            (self._translate_dimension(dim), drift) 
            for dim, drift in report.dimension_drifts.items() 
            if drift > self.drift_critical
        ]
        if critical_dims:
            dim_str = ", ".join([f"{d}({v:.2f})" for d, v in critical_dims])
            parts.append(f"\n3. 严重漂移维度: {dim_str}")
        
        parts.append(
            "\n\n请根据以上问题，重新生成响应。"
            "特别关注漂移严重的维度，确保响应符合目标人格。"
        )
        
        return "".join(parts)
    
    def get_threshold_info(self) -> Dict[str, Any]:
        """获取当前阈值配置信息"""
        return {
            'kl_threshold': self.kl_threshold,
            'drift_warning': self.drift_warning,
            'drift_critical': self.drift_critical,
            'note': 'Week 5-6: 阈值已调整为研究计划推荐值 0.70'
        }

    # ------------------------------------------------------------------ #
    # 记忆幻觉检测（HaluMem 四种失效模式增强版）
    # ------------------------------------------------------------------ #

    def check_hallucination(
        self,
        generated_text: str,
        memory_traces: List[Dict],
        speaker_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        记忆幻觉检测（覆盖 HaluMem 四种失效模式）

        失效模式：
        1. Confabulation  —— AI 编造从未存储的记忆（检测：关键事实在记忆库中无来源）
        2. Misattribution —— 把 A 说的话归给 B（检测：归因一致性）
        3. Staleness      —— 引用已被更新/覆盖的旧记忆（检测：时间顺序）
        4. Contradiction  —— 同时持有互相矛盾的记忆（检测：内容矛盾对）

        Args:
            generated_text:  要检查的生成文本
            memory_traces:   相关记忆列表（每项含 content, timestamp/created_at, trace_type 等）
            speaker_label:   当前发言人标识（用于归因检查）

        Returns:
            {
                "has_hallucination": bool,
                "risk_score": float,       # 0-1
                "issues": List[str],       # 具体问题描述
                "mode": List[str],         # 触发的失效模式
            }
        """
        issues = []
        modes  = []

        # ── 失效模式3：Staleness 时间顺序检查 ──
        staleness_issues = self._check_staleness(generated_text, memory_traces)
        if staleness_issues:
            issues.extend(staleness_issues)
            modes.append("Staleness")

        # ── 失效模式2：Misattribution 归因检查 ──
        if speaker_label is not None:
            misattrib_issues = self._check_misattribution(
                generated_text, memory_traces, speaker_label
            )
            if misattrib_issues:
                issues.extend(misattrib_issues)
                modes.append("Misattribution")

        # ── 失效模式4：Contradiction 矛盾检查 ──
        contradiction_issues = self._check_contradiction(memory_traces)
        if contradiction_issues:
            issues.extend(contradiction_issues)
            modes.append("Contradiction")

        # ── 失效模式1：Confabulation 幻构检查（轻量版：数字/日期核实）──
        confab_issues = self._check_confabulation(generated_text, memory_traces)
        if confab_issues:
            issues.extend(confab_issues)
            modes.append("Confabulation")

        # 风险评分（每个 issue 计 0.25，上限 1.0）
        risk_score = min(1.0, len(issues) * 0.25)

        return {
            "has_hallucination": len(issues) > 0,
            "risk_score": round(risk_score, 2),
            "issues": issues,
            "modes": modes,
        }

    def _check_staleness(self, text: str, traces: List[Dict]) -> List[str]:
        """
        Staleness 检查：生成文本是否引用了已被更新内容覆盖的旧记忆。
        
        策略：如果同一语义主题存在多条记忆，且较旧的那条被文本大量引用，
              而较新的条目内容与其矛盾，则标记为 Staleness。
        """
        import re, math

        def _sim(a: str, b: str) -> float:
            ta = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', a.lower())
            tb = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', b.lower())
            fa = {}
            for t in ta: fa[t] = fa.get(t, 0) + 1
            fb = {}
            for t in tb: fb[t] = fb.get(t, 0) + 1
            common = set(fa) & set(fb)
            if not common:
                return 0.0
            dot = sum(fa[t] * fb[t] for t in common)
            n1 = math.sqrt(sum(v * v for v in fa.values()))
            n2 = math.sqrt(sum(v * v for v in fb.values()))
            return dot / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0

        issues = []
        # 按创建时间排序（新 → 旧）
        sorted_traces = sorted(
            traces,
            key=lambda t: t.get('created_at', t.get('timestamp', '')) or '',
            reverse=True
        )

        for i, newer in enumerate(sorted_traces):
            newer_content = str(newer.get('content', '') or newer.get('content_summary', ''))
            if not newer_content:
                continue
            for older in sorted_traces[i+1:]:
                older_content = str(older.get('content', '') or older.get('content_summary', ''))
                if not older_content:
                    continue
                # 两条记忆主题相似（同一语义域）
                topic_sim = _sim(newer_content, older_content)
                if topic_sim < 0.30:
                    continue
                # 生成文本更接近旧记忆
                sim_to_new = _sim(text, newer_content)
                sim_to_old = _sim(text, older_content)
                if sim_to_old > sim_to_new + 0.15:
                    issues.append(
                        f"[Staleness] 文本内容（sim={sim_to_old:.2f}）更接近旧记忆而非新记忆"
                        f"（sim={sim_to_new:.2f}），可能引用了过时信息。"
                        f" 旧: {older_content[:50]}... 新: {newer_content[:50]}..."
                    )
        return issues[:2]  # 最多报2条，避免误报刷屏

    def _check_misattribution(
        self,
        text: str,
        traces: List[Dict],
        expected_speaker: str
    ) -> List[str]:
        """
        Misattribution 检查：检测生成文本中的归因是否与记忆中的发言人一致。
        
        策略：扫描文本中 "XXX说/表示/提到..." 的归因句式，
              与记忆痕迹中的发言人字段对比。
        """
        import re
        issues = []

        # 提取记忆中的已知发言人
        known_speakers = set()
        for t in traces:
            sp = t.get('speaker') or t.get('source') or t.get('trace_type', '')
            if sp:
                known_speakers.add(str(sp))

        # 扫描文本中的归因模式
        attribution_patterns = [
            r'([\u4e00-\u9fff]{2,6})(?:说|表示|提到|认为|指出)',
            r'([\u4e00-\u9fff]{2,6})(?:\'s|的)(?:意见|观点|说法)',
        ]
        for pattern in attribution_patterns:
            for match in re.finditer(pattern, text):
                attributed_to = match.group(1)
                # 如果归因到了非预期发言人（且已知记忆中有此人）
                if attributed_to != expected_speaker and attributed_to in known_speakers:
                    issues.append(
                        f"[Misattribution] 文本将内容归因于「{attributed_to}」，"
                        f"但当前发言人是「{expected_speaker}」，请核实。"
                    )
        return issues[:2]

    def _check_contradiction(self, traces: List[Dict]) -> List[str]:
        """
        Contradiction 检查：记忆库中是否存在内容互相矛盾的痕迹对。
        
        策略：寻找关键词重叠度高（同一主题）但否定词极性相反的记忆对。
        """
        import re, math

        NEGATION_WORDS = ['不', '没有', '否', '未', '非', 'not', 'no', 'never', 'without']

        def has_negation(text: str) -> bool:
            t = text.lower()
            return any(neg in t for neg in NEGATION_WORDS)

        def _sim_quick(a: str, b: str) -> float:
            ta = set(re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}', a.lower()))
            tb = set(re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}', b.lower()))
            if not ta or not tb:
                return 0.0
            return len(ta & tb) / min(len(ta), len(tb))

        issues = []
        contents = []
        for t in traces:
            c = str(t.get('content', '') or t.get('content_summary', ''))
            if c:
                contents.append(c)

        for i in range(len(contents)):
            for j in range(i + 1, len(contents)):
                sim = _sim_quick(contents[i], contents[j])
                if sim < 0.35:
                    continue
                # 相似主题但否定词极性相反
                neg_i = has_negation(contents[i])
                neg_j = has_negation(contents[j])
                if neg_i != neg_j:
                    issues.append(
                        f"[Contradiction] 记忆 {i+1} 与记忆 {j+1} 主题相似（sim={sim:.2f}）"
                        f"但否定极性相反，可能存在矛盾。"
                        f" A: {contents[i][:40]}... B: {contents[j][:40]}..."
                    )
        return issues[:2]

    def _check_confabulation(self, text: str, traces: List[Dict]) -> List[str]:
        """
        Confabulation 轻量检查：文本中的精确数字/日期是否在记忆库中有来源。
        
        策略：从生成文本中提取日期、版本号、精确数字，检查记忆库中是否有匹配。
              无来源 = 潜在幻构。
        """
        import re
        issues = []

        # 提取生成文本中的精确数值（日期 / 版本号 / 大数字）
        patterns = [
            r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',   # 日期
            r'\bv?\d+\.\d+(?:\.\d+)?\b',            # 版本号
            r'\b\d{5,}\b',                           # 5位以上大数字
        ]
        candidates = []
        for p in patterns:
            candidates.extend(re.findall(p, text))
        candidates = list(set(candidates))

        if not candidates:
            return []

        # 合并所有记忆内容为检索池
        memory_pool = ' '.join(
            str(t.get('content', '') or t.get('content_summary', ''))
            for t in traces
        )

        for c in candidates[:5]:  # 最多检查5个候选
            if c not in memory_pool:
                issues.append(
                    f"[Confabulation] 文本中的数值「{c}」在记忆库中无来源，"
                    f"请核实是否为幻构。"
                )
        return issues[:2]
