"""
E1 人格提取基线方法

BL-1: 关键词频率统计（Bag-of-Words）
"""

from typing import Dict
import re


# OCEAN 关键词词典（英文为主，中文为辅）
OCEAN_KEYWORDS = {
    'O': {
        'high': [
            # 英文（主要）
            'creative', 'curious', 'imagination', 'explore', 'novel',
            'curiosity', 'new ideas', 'experiment', 'art', 'philosophy',
            'think', 'abstract', 'complex', 'theory', 'science', 'learn',
            'try', 'experience', 'feel', 'interesting', 'amazing', 'quantum',
            'belief', 'craft', 'design', 'beauty', 'aesthetic', 'unique', 'individual',
            'open', 'innovative', 'original', 'creative', 'inquisitive', 'explorative',
            'artistic', 'philosophical', 'intellectual', 'reflective', 'analytical',
            # 中文（辅助）
            '新', '创新', '创意', '想象', '好奇', '探索', '艺术', '哲学',
            '思考', '抽象', '复杂', '理论', '科学', '学习', '尝试',
            '体验', '感受', '有趣', '神奇', '量子', '信仰', '手工艺', '设计', '美感', '审美', '独特', '个性'
        ],
        'low': [
            # 英文（主要）
            'traditional', 'conservative', 'rule', 'stable', 'routine', 'habit', 'regular',
            'standard', 'conventional', 'practical', 'actual', 'realistic', 'safe', 'familiar',
            'traditional', 'conservative', 'conventional', 'routine', 'habitual', 'predictable',
            'practical', 'down-to-earth', 'realistic', 'sensible', 'logical',
            # 中文（辅助）
            '传统', '保守', '规矩', '稳定', '按部就班', '习惯', '常规',
            '标准', '惯例', '务实', '实际', '现实', '安稳', '熟悉'
        ]
    },
    'C': {
        'high': [
            # 英文（主要）
            'plan', 'organize', 'discipline', 'punctual', 'persist', 'complete', 'task',
            'goal', 'arrange', 'list', 'organize', 'tidy', 'efficient', 'serious',
            'responsible', 'reliable', 'commitment', 'habit', 'early', 'run', 'study',
            'manage', 'record', 'track', 'execute', 'achieve',
            'organized', 'systematic', 'methodical', 'disciplined', 'responsible',
            'reliable', 'dependable', 'punctual', 'thorough', 'detail-oriented',
            'goal-oriented', 'ambitious', 'hardworking', 'diligent', 'persistent',
            # 中文（辅助）
            '计划', '组织', '自律', '准时', '坚持', '完成', '任务',
            '目标', '安排', '清单', '整理', '整洁', '效率', '认真',
            '负责', '可靠', '承诺', '习惯', '早起', '跑步', '学习',
            '安排', '管理', '记录', '跟踪', '执行', '达成'
        ],
        'low': [
            # 英文（主要）
            'casual', 'procrastinate', 'lazy', 'random', 'whatever', 'almost', 'later',
            'impulsive', 'capricious', 'disorganized', 'messy', 'forget', 'late',
            'disorganized', 'messy', 'chaotic', 'impulsive', 'spontaneous', 'careless',
            'unreliable', 'irresponsible', 'lazy', 'procrastinating', 'undisciplined',
            'unmotivated', 'aimless', 'unfocused',
            # 中文（辅助）
            '随意', '拖延', '懒', '随便', '无所谓', '差不多', '到时候再说',
            '随心', '冲动', '任性', '散漫', '混乱', '忘记', '迟到'
        ]
    },
    'E': {
        'high': [
            # 英文（主要）
            'friend', 'social', 'chat', 'party', 'meet', 'communicate', 'share',
            'lively', 'active', 'outgoing', 'talkative', 'initiative', 'enthusiastic', 'extrovert',
            'crowd', 'activity', 'organize', 'speak', 'speech', 'host', 'sing',
            'party', 'team', 'meet people', 'talk',
            'outgoing', 'sociable', 'friendly', 'gregarious', 'talkative', 'assertive',
            'energetic', 'enthusiastic', 'active', 'lively', 'spontaneous', 'adventurous',
            'confident', 'bold', 'outspoken',
            # 中文（辅助）
            '朋友', '社交', '聊天', '聚会', '认识', '交流', '分享',
            '热闹', '活泼', '开朗', '健谈', '主动', '热情', '外向',
            '人多', '活动', '组织', '发言', '演讲', '主持', 'K歌',
            '派对', '团建', '认识人', '搭话'
        ],
        'low': [
            # 英文（主要）
            'quiet', 'alone', 'introvert', 'silent', 'lonely', 'quiet',
            'shy', 'slow', 'quiet', 'dislike social', 'home', 'rest',
            'silent', 'reserved',
            'introverted', 'reserved', 'shy', 'quiet', 'withdrawn', 'inhibited',
            'reserved', 'private', 'introspective', 'thoughtful', 'contemplative',
            'calm', 'quiet', 'peaceful', 'solitude-seeking',
            # 中文（辅助）
            '安静', '独处', '内向', '沉默', '一个人', '独处', '安静',
            '社恐', '慢热', '不爱说话', '不喜欢社交', '回家', '休息',
            '沉默寡言', '不爱表达'
        ]
    },
    'A': {
        'high': [
            # 英文（主要）
            'understand', 'tolerant', 'gentle', 'kind', 'sympathy', 'care', 'help',
            'considerate', 'forgiving', 'yield', 'kind', 'friendly', 'cooperate', 'harmony',
            'no conflict', 'modest', 'listen', 'patient', 'peaceful', 'generous', 'uncalculating',
            'lend', 'give', 'let', 'help', 'support', 'comfort', 'understand',
            'pick up', 'show the way', 'carry', 'help', 'assist', 'aid', 'support',
            'helpful', 'kind', 'nice', 'good', 'friendly', 'generous', 'compassionate',
            'caring', 'thoughtful', 'considerate', 'helping', 'assisting', 'aiding',
            'kind', 'friendly', 'compassionate', 'sympathetic', 'empathetic', 'caring',
            'cooperative', 'collaborative', 'team-oriented', 'agreeable', 'harmonious',
            'patient', 'tolerant', 'forgiving', 'understanding', 'generous', 'altruistic',
            # 中文（辅助）
            '理解', '包容', '温和', '善良', '同情', '关心', '帮助',
            '体贴', '宽容', '退让', '和善', '友好', '合作', '和谐',
            '不争', '谦让', '倾听', '耐心', '和气', '大方', '不计较',
            '借', '给', '让', '帮忙', '支持', '安慰', '理解'
        ],
        'low': [
            # 英文（主要）
            'direct', 'sharp', 'pointed', 'rude', 'criticize', 'blame', 'argue',
            'grab', 'selfish', 'calculate', 'uncompromising', 'confront', 'conflict', 'tough',
            'intolerant', 'ruthless',
            'aggressive', 'assertive', 'competitive', 'dominant', 'forceful',
            'selfish', 'self-centered', 'egotistical', 'narcissistic',
            'critical', 'judgmental', 'harsh', 'unforgiving', 'intolerant',
            'argumentative', 'combative', 'hostile', 'aggressive',
            # 中文（辅助）
            '直接', '犀利', '尖锐', '不客气', '批评', '指责', '争',
            '抢', '自私', '计较', '不妥协', '对抗', '冲突', '强硬',
            '不容忍', '不留情面'
        ]
    },
    'N': {
        'high': [
            # 英文（主要）
            'anxious', 'worry', 'fear', 'nervous', 'unsettled', 'stress', 'insomnia',
            'tangled', 'sensitive', 'easily', 'collapse', 'emotion', 'trouble', 'overthink',
            'ruminate', 'repeated', 'unsettled', 'panic', 'worry', 'afraid',
            'anxious', 'sleepless', 'nervous', 'lost', 'depressed',
            'anxious', 'nervous', 'worried', 'fearful', 'tense', 'stressed',
            'anxious', 'worried', 'nervous', 'fearful', 'tense', 'stressed',
            'emotional', 'moody', 'irritable', 'sensitive', 'reactive',
            'insecure', 'self-conscious', 'self-doubting', 'hesitant',
            # 中文（辅助）
            '焦虑', '担心', '害怕', '紧张', '不安', '压力', '失眠',
            '纠结', '敏感', '容易', '崩溃', '绪', '烦恼', '多虑',
            '胡思乱想', '反复', '不安', '恐慌', '担心', '怕',
            '焦虑', '睡不着', '紧张', '失落', '沮丧'
        ],
        'low': [
            # 英文（主要）
            'calm', 'peaceful', 'cool', 'relax', 'stable', 'gentle', 'composed',
            'calm', 'whatever', 'it\'s okay', 'no hurry', 'relax', 'steady',
            'calm', 'composed', 'relaxed', 'peaceful', 'serene', 'tranquil',
            'emotionally stable', 'even-tempered', 'level-headed', 'balanced',
            'confident', 'secure', 'self-assured', 'calm', 'collected',
            'resilient', 'adaptable', 'flexible', 'calm', 'unflappable',
            # 中文（辅助）
            '淡定', '平静', '冷静', '放松', '稳定', '平和', '从容',
            '淡定', '无所谓', '没关系', '不用急', '放宽心', '稳'
        ]
    }
}


class KeywordPersonaExtractor:
    """
    基于关键词频率的人格提取器（BL-1 基线）
    
    方法：统计 OCEAN 各维度关键词出现次数
    """
    
    def __init__(self):
        self.keywords = OCEAN_KEYWORDS
    
    def extract(self, text: str) -> Dict[str, float]:
        """
        从文本中提取 OCEAN 分数
        
        Args:
            text: 输入文本
            
        Returns:
            OCEAN 五维分数字典 (0.0-1.0)
        """
        text = text.lower()
        
        scores = {}
        for dim in 'OCEAN':
            scores[dim] = self._calculate_score(text, dim)
        
        return scores
    
    def _calculate_score(self, text: str, dimension: str) -> float:
        """计算某维度的分数"""
        high_words = self.keywords[dimension]['high']
        low_words = self.keywords[dimension]['low']
        
        high_count = sum(1 for word in high_words if word in text)
        low_count = sum(1 for word in low_words if word in text)
        
        total = high_count + low_count
        if total == 0:
            return 0.5  # 无关键词时返回中性值
        
        # 计算分数：高关键词多则分数高
        score = high_count / total
        
        # 添加基础偏移，避免极端
        score = 0.3 + score * 0.4
        
        return min(1.0, max(0.0, score))
    
    def extract_with_confidence(self, text: str) -> Dict[str, float]:
        """带置信度的提取"""
        scores = self.extract(text)
        
        # 置信度基于关键词覆盖度
        text = text.lower()
        total_keywords = sum(
            len(self.keywords[d]['high']) + len(self.keywords[d]['low'])
            for d in 'OCEAN'
        )
        
        matched = 0
        for dim in 'OCEAN':
            matched += sum(1 for w in self.keywords[dim]['high'] if w in text)
            matched += sum(1 for w in self.keywords[dim]['low'] if w in text)
        
        confidence = matched / total_keywords
        
        return {
            'scores': scores,
            'confidence': min(1.0, confidence * 5)  # 放大置信度
        }


# 全局实例
BASELINE_EXTRACTOR = KeywordPersonaExtractor()
