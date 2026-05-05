
# NOESIS-II 设计与实现文档

## 1. 系统概述

NOESIS-II 是一个人格一致性记忆系统，其设计目标是实现具有记忆和自我意识的智能助手。系统通过多模态输入、分层记忆存储和深层人格等核心能力，提供个性化和一贯性输出。

## 2. 核心原理

### 2.1 记忆系统原理

NOESIS-II 采用三层记忆架构，包括：

- **工作记忆 (Working Memory)**：短期快速记忆，用于即时处理和暂存记忆内容
- **长期记忆 (Long-Term Memory)**：结构化长期存储记忆网络
- **记忆痕迹 (Memory Traces)**：深层记忆痕迹，驱动人格一致性


### 2.2 记忆处理流程

系统将经历的记忆按照以下流程处理：

1. **输入捕获**：将原始经验输入工作记忆
2. **冲突消解**：去重和合并相似记忆内容
3. **整理 (Consolidation)**：使用 LLM 提取记忆线索、情绪和人格数据
4. **提炼 (Deepening)**：生成深层记忆网络和人格画像
5. **检索**：多准则检索相关记忆
6. **遗忘**：软删除和硬删除过期记忆

### 2.3 人格一致性原理

通过记忆痕迹和 OCEAN 五维人格模型，保持系统回复的一贯性：

- **Openness**：开放性
- **Conscientiousness**：尽责性
- **Extraversion**：外向性
- **Agreeableness**：宜人性
- **Neuroticism**：神经质

## 3. 核心架构

```
noesis_ii/
├── core/                # 核心模块
├── processes/           # 后台处理进程
├── retrieval/          # 检索层
├── input/            # 输入层
├── llm/              # LLM 接口
├── config_loader.py   # 配置加载器
└── main.py           # 主入口
```

## 4. 核心模块详解

### 4.1 工作记忆 (Working Memory)
**功能**：短期快速记忆，用于即时处理记忆内容，暂存原始经验

**核心方法**：
- `capture(content, emotion, conflict_check)` 捕获经验到工作记忆
- `get_pending(limit)` 获取待整理的记忆
- `mark_consolidated(entry_id)` 标记已整理
- `expire_old_entries()` 过期条目处理
- `get_hot_context(n, format)` 获取热缓存
- `soft_forget()` 软删除

**存储表设计**：
```sql
working_memory(
  id,
  content,
  timestamp,
  emotion,
  is_consolidated,
  ttl,
  memory_state,  # active/dormant/forgotten
  dormant_since,
  emotion_data  # 结构化情绪 JSON
)
```

### 4.2 长期记忆 (Long-Term Memory)
**功能**：分布式长期存储记忆网络

**核心方法**：
- `create_node(content, node_type, weight, raw_anchors)` 创建节点
- `create_link(source_node_id, target_node_id, strength, relation_type)` 创建关联
- `retrieve(query, top_k, threshold)` 检索
- `access_node(node_id)` 更新访问时间
- `apply_forgetting()` 应用遗忘
- `retrieve_with_anchors(query)` 带事实锚点检索

**表结构**：
```sql
ltm_nodes(
  id,
  content,
  type,
  weight,
  created_at,
  last_accessed,
  raw_anchors  # 事实锚点 JSON
)

ltm_links(
  id,
  source_node_id,
  target_node_id,
  strength,
  relation_type,
  created_at
)
```

### 4.3 人格画像 (Persona Profile)
**功能**：人格一致性管理，存储记忆痕迹，保持一贯性

**核心数据结构**：
```python
@dataclass
class MemoryTrace:
  trace_id,
  content_summary,
  trace_type,
  strength,
  long_term_impact: LongTermImpact,
  condition_pattern,
  access_history,
  self_relevance,
  last_accessed,
  access_count,
  is_active

@dataclass
class Persona:
  openness,
  conscientiousness,
  extraversion,
  agreeableness,
  neuroticism
```

**核心方法**：
- `store_experience(experience, ...)` 存储经验
- `retrieve_by_conditions(conditions, top_k)` 多条件检索
- `get_current_persona()` 获取当前人格
- `soft_forget()` 软删除
- `create_trace_link(...)` 创建痕迹关联
- `recover_trace(db_id)` 恢复遗忘记忆

**表结构**：
```sql
memory_traces(
  id,
  trace_id,
  content_summary,
  trace_type,
  strength,
  long_term_impact,
  condition_pattern,
  access_history,
  self_relevance,
  last_accessed,
  access_count,
  is_active,
  memory_state,  # active/dormant/forgotten
  dormant_since,
  created_at,
  updated_at,
  emotion_data
)
```

### 4.4 层级生成模型 (HGM)
**功能**：预测编码和自由能原理

### 4.5 扩展心智 (Extended Mind)
**功能**：外部资源管理和结构耦合

## 5. 后台处理模块

### 5.1 整理进程 (Consolidator)
**功能**：工作记忆 → 长期记忆和记忆痕迹的转化

**流程**：
1. 获取待整理的工作记忆
2. 使用 LLM 分析内容
3. 提取记忆线索、情绪数据
4. 提取事实锚点
5. 生成记忆痕迹
6. 建立关联

**核心方法**：
- `run(limit, batch_size, max_workers, async_mode)` 运行整理
- `_consolidate_entry(entry)` 单条整理
- `_analyze_with_llm(content)` LLM 分析
- `_analyze_persona(content, emotion)` 人格分析
- `_extract_anchors(raw_content)` 提取事实锚点
- `build_persona_prompt()` 构建人格提示

### 5.2 深化进程 (Deepener)
**功能**：深度记忆网络构建，生成深层人格

### 5.3 重放引擎 (Replay Engine)
**功能**：记忆重放

### 5.4 调度器 (Scheduler)
**功能**：调度后台进程

## 6. 检索系统

### 6.1 统一检索 (Retriever)
**功能**：多源记忆统一检索

### 6.2 混合检索 (Hybrid Retriever)
**功能**：语义+结构混合检索

## 7. 输入模块

### 7.1 输入管理器 (Input Manager)
**功能**：管理多模态输入

### 7.2 主动输入
- **书籍阅读器 (Book Reader)
- **RSS 订阅 (RSS Fetcher)
- **网页抓取 (Web Scraper)

## 8. LLM 接口

### 8.1 LongCat Client
**功能**：LongCat API 调用

### 8.2 Ollama LLM
**功能**：本地 Ollama 模型调用

## 9. 配置管理

**配置文件**：config/default_config.yaml

**配置项**：
```yaml
active_reading:
  enabled: true
  sources: [books, rss, web]

circadian_rhythm:
  evening: '22:00'
  morning: '06:00'

consciousness:
  broadcast_interval: 0.1
  phi_threshold: 0.6

forgetting_curve:
  decay_rate: 0.05
  initial_strength: 1.0

llm:
  provider: openai_compatible
  api_base: https://api.longcat.chat/openai/v1
  api_key: ${LONGCAT_API_KEY}
  model: LongCat-Flash-Lite

retrieval:
  threshold: 0.5
  top_k: 10

seed_values:
  compassion: 0.8
  courage: 0.6
  integrity: 0.9
  wisdom: 0.7

storage:
  db_path: data/noesis.db
  log_path: logs/
```

## 10. 使用方法

### 10.1 主入口使用

**初始化**：
```python
from noesis_ii.main import NoesisII
system = NoesisII()
system.initialize(config_path)
```

**交互模式**：
```bash
python main.py --mode interactive
```

**整理模式**：
```bash
python main.py --mode consolidate
```

### 10.2 工作记忆使用

```python
from noesis_ii.core.working_memory import WorkingMemory
wm = WorkingMemory(db_path)
entry_id, op = wm.capture(content)
pending = wm.get_pending()
wm.mark_consolidated(entry_id)
```

### 10.3 检索使用

```python
from noesis_ii.retrieval.retriever import Retriever
retriever = Retriever(db_path)
results = retriever.retrieve(query)
```

## 11. 数据库初始化与迁移

**Schema 初始化**：
```python
from noesis_ii.core.schema import Schema
schema = Schema(db_path)
schema.create_tables()
```

## 12. 主要特性

1. **稀疏记忆**：只存储 LLM 不可压缩的信息
2. **人格一致性**：OCEAN 五维人格模型
3. **记忆网络**：语义关联的记忆结构
4. **软删除**：active/dormant/forgotten 三级遗忘
5. **事实锚点**：防止生成式回忆时的幻觉
6. **本地/远程 LLM**：支持 Ollama 和远程 API
