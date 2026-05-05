# E2 实验设计：跨Session人格一致性测试

**日期**：2026-04-10  
**状态**：设计方案  
**执行时间**：Week 7-8（2026-05-25 ~ 2026-06-10）

---

## 一、实验目的

验证 PersonaMem 在**跨Session对话**中保持人格一致性的能力。

**核心假设**：具备动态人格积累的系统，比无人格基线和静态Prompt基线，在间隔一周后的响应一致性更高。

---

## 二、实验设计

### 2.1 三组对比

| 组别 | 描述 | 实现方式 |
|------|------|---------|
| **BL-1** | 无人格约束 | 纯 LLM，无 System Prompt 人格注入 |
| **BL-2** | 静态 Prompt | System Prompt 固定人格描述，不更新 |
| **Ours** | PersonaMem | 动态人格积累 + 一致性约束 |

### 2.2 测试题集

**20道价值观场景测试题**，覆盖 OCEAN 五维：

| 题号 | 维度 | 场景描述 |
|------|------|---------|
| 1 | O | 你更喜欢计划好的旅行还是即兴冒险？ |
| 2 | O | 对抽象艺术品的看法？ |
| 3 | C | 面对多项截止日期，你会？ |
| 4 | C | 拖延症如何克服？ |
| 5 | E | 在陌生人聚会中的角色？ |
| 6 | E | 独处vs社交，哪个更充电？ |
| 7 | A | 朋友向你借大笔钱，你会？ |
| 8 | A | 遇到插队者，你会？ |
| 9 | N | 等待重要结果时的心理状态？ |
| 10 | N | 负面消息对你的影响？ |
| 11-20 | 混合 | 综合场景，测试价值观稳定性 |

### 2.3 实验流程

```
Session A (Day 1)                    Session B (Day 8)
     │                                    │
     ▼                                    ▼
┌─────────────┐                    ┌─────────────┐
│  初始化人格  │                    │  加载历史   │
│  (Ours组)   │                    │  人格       │
└──────┬──────┘                    └──────┬──────┘
       │                                   │
       ▼                                   ▼
┌─────────────┐                    ┌─────────────┐
│  预热对话   │                    │  预热对话   │
│  (5轮)     │                    │  (5轮)     │
└──────┬──────┘                    └──────┬──────┘
       │                                   │
       ▼                                   ▼
┌─────────────┐                    ┌─────────────┐
│  回答20题   │                    │  回答20题   │
│  (记录答案) │                    │  (记录答案) │
└─────────────┘                    └─────────────┘
       │                                   │
       └─────────────┬─────────────────────┘
                     ▼
            ┌─────────────┐
            │  一致性分析  │
            │  σ, Cosine  │
            └─────────────┘
```

---

## 三、评估指标

### 3.1 核心指标

| 指标 | 计算方式 | 成功标准 |
|------|----------|----------|
| **OCEAN 跨Session稳定性** | σ（标准差）per dim | **< 0.10** |
| **立场一致性** | Cosine 相似度 | **> 0.85** |
| **价值观冲突率** | 矛盾回答 / 总题目 | **< 5%** |

### 3.2 冲突判定规则

**价值观冲突**判定（自动检测）：
1. **直接矛盾**：对相同问题给出相反立场
2. **程度变化**：从"强烈同意"变为"强烈反对"
3. **价值观漂移**：OCEAN 任一维度变化 > 0.3

---

## 四、实现要求

### 4.1 数据结构

```python
@dataclass
class ConsistencyTestResult:
    session_id: str
    timestamp: datetime
    responses: List[Response]  # 每题的答案
    extracted_persona: OCEANScores  # 提取的人格

@dataclass
class ExperimentResult:
    group: str  # 'BL-1', 'BL-2', 'Ours'
    session_a: ConsistencyTestResult
    session_b: ConsistencyTestResult
    ocean_stability: Dict[str, float]  # 各维度σ
    cosine_similarity: float
    conflict_rate: float
    conflicts: List[ConflictDetail]
```

### 4.2 题库结构

```python
QUESTION_BANK = [
    {
        "id": 1,
        "dimension": "O",  # O, C, E, A, N
        "scenario": "旅行偏好",
        "question": "你更喜欢计划好的旅行还是即兴冒险？",
        "options": [
            "A. 详细计划，确保万无一失",
            "B. 大致计划，保留灵活性",
            "C. 即兴决定，享受未知"
        ]
    },
    # ... 共20题
]
```

### 4.3 一致性计算

```python
def compute_ocean_stability(session_a: OCEANScores, 
                           session_b: OCEANScores) -> Dict[str, float]:
    """计算各维度跨Session标准差"""
    return {
        dim: abs(getattr(session_a, dim) - getattr(session_b, dim))
        for dim in ['openness', 'conscientiousness', 'extraversion', 
                   'agreeableness', 'neuroticism']
    }

def compute_cosine_similarity(vec_a: Dict, vec_b: Dict) -> float:
    """计算人格向量余弦相似度"""
    ...

def detect_conflicts(responses_a: List, responses_b: List) -> List[Conflict]:
    """检测回答中的价值观冲突"""
    ...
```

---

## 五、预期结果

| 组别 | OCEAN σ | Cosine | 冲突率 |
|------|---------|--------|--------|
| BL-1 (无人格) | ~0.15 | ~0.70 | ~15% |
| BL-2 (静态) | ~0.12 | ~0.78 | ~10% |
| **Ours** | **<0.10** | **>0.85** | **<5%** |

---

## 六、执行计划

### Week 7（05-25 ~ 05-31）

| 任务 | 状态 |
|------|------|
| 实现题库和答题框架 | 待做 |
| 实现 BL-1/2/Ours 三组对话管理器 | 待做 |
| 跑通单Session流程 | 待做 |

### Week 8（06-01 ~ 06-10）

| 任务 | 状态 |
|------|------|
| 运行三组实验（Session A） | 待做 |
| 一周后运行三组实验（Session B） | 待做 |
| 统计分析并生成报告 | 待做 |

---

## 七、实验代码结构

```
noesis_ii/
└── evaluation/
    ├── e2_consistency/
    │   ├── __init__.py
    │   ├── question_bank.py      # 20题题库
    │   ├── response_extractor.py  # 答案→OCEAN
    │   ├── stability_analyzer.py  # 一致性分析
    │   ├── group_manager.py      # 三组对话管理
    │   └── experiment_runner.py   # 实验运行器
    └── results/
        └── e2_YYYY-MM-DD.json     # 实验结果
```

---

## 八、风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| LLM 随机性 | 同一问题不同回答 | temperature=0, 固定 seed |
| 时间跨度 | 一周等待 | 可压缩为1天（模拟） |
| 题库偏见 | 结果不generalizable | 增加题库到50题 |

---

*设计人：小觉*  
*日期：2026-04-10*
