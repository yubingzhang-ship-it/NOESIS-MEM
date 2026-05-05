"""
层级化生成模型 (Hierarchical Generative Model, HGM)

实现自由能原理 (FEP) 的预测编码：
- 四层架构：感官层(L0) → 物体层(L1) → 情境层(L2) → 叙事层(L3)
- 自下而上的预测误差传递 + 自上而下的先验预测
- 变分自由能最小化驱动学习
- 精度加权注意力机制

依赖 NumPy（不依赖 PyTorch），使用简化的变分推断。
"""

import numpy as np
import json
import os
import sqlite3
import time
import math
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 softmax"""
    x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """sigmoid 激活"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _relu(x: np.ndarray) -> np.ndarray:
    """ReLU 激活"""
    return np.maximum(0, x)


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    计算 KL(p || q)，两个概率分布间的 KL 散度
    p, q 为归一化概率向量
    """
    eps = 1e-10
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def _cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """交叉熵 H(p, q)"""
    eps = 1e-10
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(-np.sum(p * np.log(q)))


def _text_to_token_distribution(text: str, vocab_size: int = 64) -> np.ndarray:
    """
    将文本转换为词频/字符频分布（归一化概率向量）
    用于 HGM 的输入表示
    """
    text = str(text).strip()
    if not text:
        return np.ones(vocab_size) / vocab_size

    # 构建字符频率
    freq = np.zeros(vocab_size)
    for ch in text:
        # 使用hash代替ord%64减少碰撞
        idx = hash(ch) % vocab_size
        freq[idx] += 1

    # 归一化为概率分布
    total = freq.sum()
    if total > 0:
        freq /= total
    else:
        freq = np.ones(vocab_size) / vocab_size

    return freq


def _text_to_semantic_vector(text: str, dim: int = 64) -> np.ndarray:
    """
    将文本转换为紧凑的语义特征向量
    基于字符 n-gram 统计的简化 embedding
    """
    text = str(text).lower().strip()
    if not text:
        return np.zeros(dim)

    vec = np.zeros(dim)
    # 单字特征
    for ch in text:
        idx = hash(ch) % dim
        vec[idx] += 1.0
    # 二字 bigram 特征
    for i in range(len(text) - 1):
        bigram = text[i:i+2]
        idx = (hash(bigram[0]) * 31 + hash(bigram[1])) % dim
        vec[idx] += 0.5

    # L2 归一化
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    return vec


# ═══════════════════════════════════════════════════════════════
# 单个生成层级
# ═══════════════════════════════════════════════════════════════

@dataclass
class GenerativeLevel:
    """
    单个生成层级（对应设计文档的 GenerativeLevel）

    实现简化的变分自编码器结构：
    - 编码器：推断隐藏状态 q(s|o) → μ, σ²
    - 解码器：生成预测 p(o|s) → 概率分布
    - 精度控制器：注意力机制（误差越大精度越高）
    """
    level_id: int
    state_dim: int
    obs_dim: int
    hidden_dim: int = 64

    # 后验状态（学习到的内部表征）
    posterior_mu: np.ndarray = field(default=None, repr=False)
    posterior_logvar: np.ndarray = field(default=None, repr=False)
    prediction: np.ndarray = field(default=None, repr=False)
    prior_mu: np.ndarray = field(default=None, repr=False)

    # 编码器/解码器权重（随机投影近似）
    encoder_w1: np.ndarray = field(default=None, repr=False)
    encoder_b1: np.ndarray = field(default=None, repr=False)
    encoder_w2: np.ndarray = field(default=None, repr=False)
    encoder_b2: np.ndarray = field(default=None, repr=False)
    decoder_w1: np.ndarray = field(default=None, repr=False)
    decoder_b1: np.ndarray = field(default=None, repr=False)
    decoder_w2: np.ndarray = field(default=None, repr=False)
    decoder_b2: np.ndarray = field(default=None, repr=False)

    # 精度权重
    precision: np.ndarray = field(default=None, repr=False)

    # 更新历史
    update_count: int = 0
    last_prediction_error: float = 0.0
    surprise_history: list = field(default_factory=list)

    def __post_init__(self):
        """初始化权重（Xavier 初始化）"""
        rng = np.random.RandomState(42 + self.level_id)

        # 编码器: obs_dim → hidden_dim → state_dim * 2
        scale1 = np.sqrt(2.0 / (self.obs_dim + self.hidden_dim))
        self.encoder_w1 = rng.randn(self.obs_dim, self.hidden_dim) * scale1
        self.encoder_b1 = np.zeros(self.hidden_dim)
        scale2 = np.sqrt(2.0 / (self.hidden_dim + self.state_dim * 2))
        self.encoder_w2 = rng.randn(self.hidden_dim, self.state_dim * 2) * scale2
        self.encoder_b2 = np.zeros(self.state_dim * 2)

        # 解码器: state_dim → hidden_dim → obs_dim
        scale3 = np.sqrt(2.0 / (self.state_dim + self.hidden_dim))
        self.decoder_w1 = rng.randn(self.state_dim, self.hidden_dim) * scale3
        self.decoder_b1 = np.zeros(self.hidden_dim)
        scale4 = np.sqrt(2.0 / (self.hidden_dim + self.obs_dim))
        self.decoder_w2 = rng.randn(self.hidden_dim, self.obs_dim) * scale4
        self.decoder_b2 = np.zeros(self.obs_dim)

        # 初始化精度为均匀
        self.precision = np.ones(self.obs_dim) * 0.5

        # 初始化后验为零向量
        self.posterior_mu = np.zeros(self.state_dim)
        self.posterior_logvar = np.zeros(self.state_dim)

    def encode(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """变分编码：返回 μ 和 log(σ²)"""
        h = _relu(observation @ self.encoder_w1 + self.encoder_b1)
        params = h @ self.encoder_w2 + self.encoder_b2
        mu = params[:self.state_dim]
        logvar = params[self.state_dim:]
        # 限制 logvar 范围，防止数值不稳定
        logvar = np.clip(logvar, -4.0, 4.0)
        return mu, logvar

    def decode(self, state: np.ndarray) -> np.ndarray:
        """生成预测分布 p(o|s)"""
        h = _relu(state @ self.decoder_w1 + self.decoder_b1)
        logits = h @ self.decoder_w2 + self.decoder_b2
        # 归一化为概率分布
        return _softmax(logits)

    def sample_posterior(self, rng: np.random.RandomState = None) -> np.ndarray:
        """重参数化采样：z = μ + σ * ε"""
        if rng is None:
            rng = np.random.RandomState()
        std = np.exp(0.5 * self.posterior_logvar)
        eps = rng.randn(self.state_dim)
        return self.posterior_mu + std * eps

    def compute_kl_divergence(self) -> float:
        """
        计算 KL(q(s|o) || N(0, I))
        KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        """
        kl = -0.5 * np.sum(
            1 + self.posterior_logvar
            - self.posterior_mu ** 2
            - np.exp(self.posterior_logvar)
        )
        return float(np.clip(kl, 0, 100))

    def compute_surprise(self, observation: np.ndarray) -> float:
        """
        计算惊奇度（负对数似然）
        surprise = -log p(o|s)
        """
        pred = self.prediction
        if pred is None:
            return 0.0
        eps = 1e-10
        obs_prob = np.clip(observation, eps, 1.0)
        pred_prob = np.clip(pred, eps, 1.0)
        return float(-np.sum(obs_prob * np.log(pred_prob)))

    def update_precision(self, prediction_error: np.ndarray):
        """
        精度更新（注意力机制）
        高误差区域 → 高精度 → 快速学习
        """
        error_mag = np.abs(prediction_error)
        # 指数移动平均更新
        alpha = 0.3
        self.precision = (1 - alpha) * self.precision + alpha * error_mag
        # 钳制到合理范围
        self.precision = np.clip(self.precision, 0.1, 1.0)

    def update_posterior(self, prior_prediction: np.ndarray,
                         weighted_error: np.ndarray):
        """
        贝叶斯更新：结合先验预测和加权观测误差
        近似后验 = 先验 + 加权误差 → 重新编码
        """
        combined = prior_prediction + weighted_error
        mu, logvar = self.encode(combined)

        # 与现有后验的混合（平滑更新）
        blend = 0.7
        self.posterior_mu = blend * mu + (1 - blend) * self.posterior_mu
        self.posterior_logvar = blend * logvar + (1 - blend) * self.posterior_logvar

        # 生成新预测
        sampled = self.sample_posterior()
        self.prediction = self.decode(sampled)
        self.update_count += 1


# ═══════════════════════════════════════════════════════════════
# 主模型
# ═══════════════════════════════════════════════════════════════

class HierarchicalGenerativeModel:
    """
    层级化生成模型

    实现自由能原理的预测编码：
    - 每层从上层接收先验预测
    - 与下层传入的预测误差结合
    - 生成后验信念
    - 自由能 = 精度加权预测误差 + KL散度（复杂度惩罚）
    - 学习率由唤醒度和惊奇度自适应调节

    四层结构（对应设计文档）：
    L0 感官层：边缘/颜色/声音特征 → 低级模式识别
    L1 物体层：物体识别 → "是什么"
    L2 情境层：情境理解 → "在哪里、何时"
    L3 叙事层：叙事整合 → "为什么、意味着什么"
    """

    # 层级定义（简化维度，适配 NumPy）
    # 关键约束：每层的 obs_dim 必须等于下一层的 obs_dim（用于自上而下精炼）
    LEVEL_CONFIGS = [
        {'state_dim': 32, 'obs_dim': 64,  'hidden_dim': 32},   # L0 感官层
        {'state_dim': 48, 'obs_dim': 64,  'hidden_dim': 48},   # L1 物体层
        {'state_dim': 64, 'obs_dim': 64,  'hidden_dim': 48},   # L2 情境层
        {'state_dim': 64, 'obs_dim': 64,  'hidden_dim': 48},   # L3 叙事层
    ]

    LEVEL_NAMES = ['感官层', '物体层', '情境层', '叙事层']

    def __init__(self, db_path: str = None, config: Dict = None):
        self.db_path = db_path
        self.config = config or {}
        self.levels: List[GenerativeLevel] = []
        self.update_history: List[Dict] = []
        self.total_free_energy = 0.0
        self.rng = np.random.RandomState(42)
        self._db_initialized = False

        # 初始化四层
        for i, cfg in enumerate(self.LEVEL_CONFIGS):
            level = GenerativeLevel(
                level_id=i,
                state_dim=cfg['state_dim'],
                obs_dim=cfg['obs_dim'],
                hidden_dim=cfg['hidden_dim']
            )
            self.levels.append(level)

        # 加载已保存的模型状态
        if db_path:
            self._load_state()

    # ─────────────────────────────────────────────────────
    # 核心处理
    # ─────────────────────────────────────────────────────

    def process_observation(self, observation: str,
                            emotion_intensity: float = 0.5) -> Dict:
        """
        处理观察：完整的预测编码周期

        对应神经科学的：
        - 腹侧流（物体识别）：L0→L1→L2
        - 背侧流（空间动作）：L2→动作
        - 默认网络（叙事）：L3整合

        返回包含预测误差、惊奇度、自由能等的完整结果
        """
        # 将文本观察转为概率分布
        obs_vector = _text_to_token_distribution(observation)

        # === 第一步：自下而上的预测误差计算 ===
        prediction_errors = []
        surprise_values = []
        current_input = obs_vector

        for i, level in enumerate(self.levels):
            # 生成预测（来自上层的先验）
            if level.prediction is None:
                level.prediction = level.decode(np.zeros(level.state_dim))

            # 计算预测误差
            if level.level_id == 0:
                pred_error = current_input - level.prediction
            else:
                prev_prediction = self.levels[level.level_id - 1].prediction
                if prev_prediction is not None and level.prediction is not None:
                    pred_error = prev_prediction - level.prediction
                else:
                    pred_error = np.zeros(level.obs_dim)

            error_magnitude = float(np.linalg.norm(pred_error))

            # 精度加权（注意力机制）
            level.update_precision(pred_error)
            weighted_error = pred_error * level.precision

            # 更新后验（贝叶斯更新近似）
            level.update_posterior(level.prediction, weighted_error)

            # 记录惊奇度
            surprise = level.compute_surprise(current_input)
            level.last_prediction_error = error_magnitude
            level.surprise_history.append(surprise)

            prediction_errors.append({
                'level': level.level_id,
                'level_name': self.LEVEL_NAMES[i],
                'error_magnitude': round(error_magnitude, 6),
                'precision_mean': round(float(np.mean(level.precision)), 4),
                'surprise': round(surprise, 6),
            })
            surprise_values.append(surprise)

            # 传递到下一层
            if i < len(self.levels) - 1:
                current_input = level.prediction
        
        # 限制惊奇历史长度
        for level in self.levels:
            if len(level.surprise_history) > 100:
                level.surprise_history = level.surprise_history[-50:]

        # === 第二步：自上而下的预测精炼 ===
        for i in range(len(self.levels) - 1, 0, -1):
            high_level = self.levels[i]
            low_level = self.levels[i - 1]
            high_prior = high_level.decode(high_level.posterior_mu)
            # 将高层先验作为低层的额外输入进行精炼
            refined = low_level.prediction * 0.7 + high_prior[:low_level.obs_dim] * 0.3
            low_level.prediction = refined

        # === 第三步：计算总自由能 ===
        total_surprise = sum(surprise_values)
        total_kl = sum(l.compute_kl_divergence() for l in self.levels)
        # 自由能 = 惊奇 + 复杂度惩罚
        free_energy = total_surprise + 0.1 * total_kl
        self.total_free_energy = free_energy

        # 高惊奇 → 可进入意识（传给 IAB）
        consciousness_eligible = total_surprise > 0.5

        result = {
            'total_surprise': round(total_surprise, 6),
            'total_kl': round(total_kl, 6),
            'free_energy': round(free_energy, 6),
            'consciousness_eligible': consciousness_eligible,
            'prediction_errors': prediction_errors,
            'level_states': {
                self.LEVEL_NAMES[i]: {
                    'state_norm': round(float(np.linalg.norm(l.posterior_mu)), 4),
                    'kl_divergence': round(l.compute_kl_divergence(), 4),
                    'update_count': l.update_count,
                } for i, l in enumerate(self.levels)
            }
        }

        # 记录更新历史
        self.update_history.append({
            'timestamp': time.time(),
            'free_energy': free_energy,
            'total_surprise': total_surprise,
            'consciousness_eligible': consciousness_eligible,
        })
        
        # 限制历史记录长度
        if len(self.update_history) > 1000:
            self.update_history = self.update_history[-500:]

        return result

    def imagine(self, context: str, steps: int = 5) -> List[Dict]:
        """
        想象/模拟：脱离感官输入，纯生成过程（只读模拟，不修改模型状态）
        
        对应：心理模拟、规划、梦境
        自上而下生成预测（无感官约束）
        """
        # 保存当前状态，模拟后恢复
        saved_states = [(l.posterior_mu.copy(), l.posterior_logvar.copy(), 
                         l.prediction.copy() if l.prediction is not None else None)
                        for l in self.levels]

        try:
            # 用上下文初始化最高层
            if context:
                context_vec = _text_to_semantic_vector(context, self.levels[-1].state_dim)
                self.levels[-1].posterior_mu = context_vec

            imagined_trajectory = []

            for step in range(steps):
                step_result = {}

                # 自上而下生成
                for i in range(len(self.levels) - 1, -1, -1):
                    level = self.levels[i]
                    # 采样后验并解码
                    sampled = level.sample_posterior(self.rng)
                    level.prediction = level.decode(sampled)
                    step_result[self.LEVEL_NAMES[i]] = {
                        'state_norm': round(float(np.linalg.norm(sampled)), 4),
                    }

                    # 将高层输出传递给低层作为先验
                    if i > 0:
                        lower = self.levels[i - 1]
                        prior = level.prediction[:lower.state_dim]
                        lower.posterior_mu = lower.posterior_mu * 0.8 + prior * 0.2

                step_result['step'] = step
                step_result['timestamp'] = time.time()
                imagined_trajectory.append(step_result)

                # 最高层动态演化（模拟时间推进）
                top = self.levels[-1]
                noise = self.rng.randn(top.state_dim) * 0.1
                top.posterior_mu = top.posterior_mu + noise
                # 添加轻微衰减（回归倾向）
                top.posterior_mu *= 0.95

            return imagined_trajectory
        finally:
            # 恢复状态
            for l, (mu, logvar, pred) in zip(self.levels, saved_states):
                l.posterior_mu = mu
                l.posterior_logvar = logvar
                l.prediction = pred

    def consolidate(self, observation: str,
                    emotion_intensity: float = 0.5,
                    learning_rate: float = None) -> Dict:
        """
        巩固：更新生成模型参数

        这是"记忆形成"的真正机制：自由能最小化
        使用梯度下降近似更新编码器/解码器权重

        对应神经科学：突触可塑性（LTP/LTD）
        """
        obs_vector = _text_to_token_distribution(observation)

        # 处理观察（计算预测误差）
        process_result = self.process_observation(observation, emotion_intensity)

        # 自适应学习率（唤醒度调节）
        if learning_rate is None:
            learning_rate = self._adaptive_lr(emotion_intensity, process_result['total_surprise'])

        # === 梯度下降更新权重 ===
        total_param_change = 0.0

        for level in self.levels:
            if level.prediction is None:
                continue

            # 计算预测误差梯度（简化）
            if level.level_id == 0:
                target = obs_vector
            else:
                target = self.levels[level.level_id - 1].prediction[:level.obs_dim] if self.levels[level.level_id - 1].prediction is not None else np.zeros(level.obs_dim)

            error = target - level.prediction
            param_change = self._update_weights(level, error, learning_rate)
            total_param_change += param_change

        # 记录更新
        self.update_history.append({
            'timestamp': time.time(),
            'type': 'consolidation',
            'surprise': process_result['total_surprise'],
            'learning_rate': learning_rate,
            'parameter_change': round(total_param_change, 6),
            'emotion_intensity': emotion_intensity,
        })

        # 持久化
        self._save_state()

        return {
            'free_energy': process_result['free_energy'],
            'surprise': process_result['total_surprise'],
            'learning_rate': round(learning_rate, 6),
            'parameter_change': round(total_param_change, 6),
        }

    def _adaptive_lr(self, arousal: float, surprise: float) -> float:
        """
        适应性学习率：高唤醒高惊奇 = 快速学习

        对应神经科学的去甲肾上腺素调节：
        - 蓝斑核释放去甲肾上腺素
        - 高 NE → 高可塑性 → 快速学习
        - 低 NE → 稳定 → 慢速学习
        """
        base_lr = 0.001
        # 唤醒度 boost (1-2x)
        arousal_boost = 1.0 + arousal
        # 惊奇度 boost (1-2x)
        surprise_boost = 1.0 + min(surprise, 2.0) / 2.0

        return base_lr * arousal_boost * surprise_boost

    def _update_weights(self, level: GenerativeLevel,
                        error: np.ndarray, lr: float) -> float:
        """
        简化的梯度下降权重更新

        对真实 VAE 训练的近似：
        - 使用预测误差作为梯度的代理
        - 仅更新解码器权重（编码器权重更新更复杂，暂保持稳定）
        """
        change = 0.0

        # 解码器权重更新（基于预测误差）
        sampled = level.sample_posterior(self.rng)
        h = _relu(sampled @ level.decoder_w1 + level.decoder_b1)

        # 输出层梯度
        grad_w2 = np.outer(h, error * level.precision) * lr
        grad_b2 = error * level.precision * lr

        # 隐藏层梯度（简化的反向传播）
        grad_h = (error * level.precision) @ level.decoder_w2.T
        grad_h = grad_h * (h > 0).astype(float)  # ReLU 导数
        grad_w1 = np.outer(sampled, grad_h) * lr
        grad_b1 = grad_h * lr

        # 梯度裁剪，防止梯度爆炸
        max_grad_norm = 5.0
        grad_w2_norm = np.linalg.norm(grad_w2)
        if grad_w2_norm > max_grad_norm:
            grad_w2 = grad_w2 * max_grad_norm / grad_w2_norm
        grad_w1_norm = np.linalg.norm(grad_w1)
        if grad_w1_norm > max_grad_norm:
            grad_w1 = grad_w1 * max_grad_norm / grad_w1_norm
        
        # 应用更新
        level.decoder_w2 += grad_w2
        level.decoder_b2 += grad_b2
        level.decoder_w1 += grad_w1
        level.decoder_b1 += grad_b1

        change = float(np.sum(np.abs(grad_w2)) + np.sum(np.abs(grad_w1)))

        # 重新生成预测
        level.prediction = level.decode(sampled)

        return change

    # ─────────────────────────────────────────────────────
    # 查询与统计
    # ─────────────────────────────────────────────────────

    def get_free_energy(self) -> float:
        """获取当前总自由能"""
        return self.total_free_energy

    def get_level_states(self) -> Dict:
        """获取各层状态摘要"""
        states = {}
        for i, level in enumerate(self.levels):
            states[self.LEVEL_NAMES[i]] = {
                'update_count': level.update_count,
                'kl_divergence': round(level.compute_kl_divergence(), 4),
                'precision_mean': round(float(np.mean(level.precision)), 4),
                'state_norm': round(float(np.linalg.norm(level.posterior_mu)), 4),
                'avg_surprise': round(
                    float(np.mean(level.surprise_history[-10:])) if level.surprise_history else 0.0, 4
                ),
            }
        return states

    def get_stats(self) -> Dict:
        """获取 HGM 统计信息"""
        return {
            'total_updates': sum(l.update_count for l in self.levels),
            'total_free_energy': round(self.total_free_energy, 4),
            'update_history_count': len(self.update_history),
            'level_states': self.get_level_states(),
        }

    # ─────────────────────────────────────────────────────
    # 持久化
    # ─────────────────────────────────────────────────────

    def _init_db(self):
        """初始化 HGM 相关数据库表"""
        if not self.db_path:
            return
        dir_name = os.path.dirname(self.db_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=10)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hgm_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hgm_update_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                free_energy REAL,
                total_surprise REAL,
                learning_rate REAL,
                parameter_change REAL,
                emotion_intensity REAL,
                timestamp TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    def _save_state(self):
        """保存模型状态到数据库"""
        if not self.db_path:
            return
        if not self._db_initialized:
            self._init_db()
            self._db_initialized = True
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            state = {
                'levels': [],
                'total_free_energy': self.total_free_energy,
            }
            for level in self.levels:
                level_state = {
                    'level_id': level.level_id,
                    'posterior_mu': level.posterior_mu.tolist(),
                    'posterior_logvar': level.posterior_logvar.tolist(),
                    'precision': level.precision.tolist(),
                    'update_count': level.update_count,
                    'encoder_w1': level.encoder_w1.tolist(),
                    'encoder_b1': level.encoder_b1.tolist(),
                    'encoder_w2': level.encoder_w2.tolist(),
                    'encoder_b2': level.encoder_b2.tolist(),
                    'decoder_w1': level.decoder_w1.tolist(),
                    'decoder_b1': level.decoder_b1.tolist(),
                    'decoder_w2': level.decoder_w2.tolist(),
                    'decoder_b2': level.decoder_b2.tolist(),
                }
                state['levels'].append(level_state)

            state_json = json.dumps(state, ensure_ascii=False)
            now = time.strftime('%Y-%m-%d %H:%M:%S')

            cursor.execute('''
                INSERT INTO hgm_state (id, state_json, updated_at) VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET state_json = excluded.state_json, updated_at = excluded.updated_at
            ''', (state_json, now))

            conn.commit()
        finally:
            conn.close()

    def _load_state(self):
        """从数据库加载模型状态"""
        if not self.db_path:
            return
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT state_json FROM hgm_state WHERE id = 1')
            row = cursor.fetchone()
            conn.close()
            if not row:
                return

            state = json.loads(row[0])
            self.total_free_energy = state.get('total_free_energy', 0.0)

            for level_data in state.get('levels', []):
                level_id = level_data['level_id']
                if level_id < len(self.levels):
                    level = self.levels[level_id]
                    level.posterior_mu = np.array(level_data['posterior_mu'])
                    level.posterior_logvar = np.array(level_data['posterior_logvar'])
                    level.precision = np.array(level_data['precision'])
                    level.update_count = level_data.get('update_count', 0)
                    level.encoder_w1 = np.array(level_data['encoder_w1'])
                    level.encoder_b1 = np.array(level_data['encoder_b1'])
                    level.encoder_w2 = np.array(level_data['encoder_w2'])
                    level.encoder_b2 = np.array(level_data['encoder_b2'])
                    level.decoder_w1 = np.array(level_data['decoder_w1'])
                    level.decoder_b1 = np.array(level_data['decoder_b1'])
                    level.decoder_w2 = np.array(level_data['decoder_w2'])
                    level.decoder_b2 = np.array(level_data['decoder_b2'])
                    level.prediction = level.decode(level.posterior_mu)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            print(f"[HGM] Failed to load state: {e}")

    def log_update(self, result: Dict):
        """记录更新日志"""
        if not self.db_path:
            return
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO hgm_update_log
                (free_energy, total_surprise, learning_rate, parameter_change, emotion_intensity, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                result.get('free_energy', 0),
                result.get('surprise', 0),
                result.get('learning_rate', 0),
                result.get('parameter_change', 0),
                result.get('emotion_intensity', 0),
                time.strftime('%Y-%m-%d %H:%M:%S'),
            ))
            conn.commit()
        finally:
            conn.close()
