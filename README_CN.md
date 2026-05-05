# NOESIS-II 人格一致性记忆系统

> 具有记忆和自我意识的智能助手系统

---

## 1. 系统概述

NOESIS-II 是一个人格一致性记忆系统，设计目标是实现具有记忆和自我意识的智能助手。系统通过多模态输入、分层记忆存储和深层人格等核心能力，提供个性化和一贯性输出。

### 核心原理

系统采用三层记忆架构：

- **工作记忆 (Working Memory)**：短期快速记忆，用于即时处理和暂存记忆内容
- **长期记忆 (Long-Term Memory)**：结构化长期存储记忆网络
- **记忆痕迹 (Memory Traces)**：深层记忆痕迹，驱动人格一致性

### 记忆处理流程

1. **输入捕获**：将原始经验输入工作记忆
2. **冲突消解**：去重和合并相似记忆内容
3. **整理 (Consolidation)**：使用 LLM 提取记忆线索、情绪和人格数据
4. **提炼 (Deepening)**：生成深层记忆网络和人格画像
5. **检索**：多准则检索相关记忆
6. **遗忘**：软删除和硬删除过期记忆

---

## 2. 主要特性

1. **稀疏记忆**：只存储 LLM 不可压缩的信息
2. **人格一致性**：OCEAN 五维人格模型（开放性、尽责性、外向性、宜人性、神经质）
3. **记忆网络**：语义关联的记忆结构
4. **软删除**：active/dormant/forgotten 三级遗忘机制
5. **事实锚点**：防止生成式回忆时的幻觉
6. **多 LLM 支持**：支持 Ollama 本地模型和远程 API

---

## 3. 环境要求与安装

### 环境要求

- Python 3.10+
- Windows / Linux / macOS
- LLM API 访问权限或本地 Ollama 安装

### 依赖包 (requirements.txt)

```
python-dotenv>=1.0.0
PyYAML>=6.0
schedule>=1.2.0
sqlalchemy>=2.0.0
requests>=2.31.0
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
gradio>=4.0.0
ollama>=0.1.0
openai>=1.0.0
```

### 安装步骤

```bash
# 安装依赖
pip install -r requirements.txt
```

---

## 4. 快速开始

```bash
# 进入 noesis_ii 目录
cd noesis_ii

# 交互模式
python main.py

# 指定配置文件
python main.py --config config/default_config.yaml

# 守护进程模式
python main.py --mode daemon

# 执行一次记忆巩固
python main.py --mode consolidate

# 执行一次深度分析
python main.py --mode deepen
```

---

## 5. 核心架构

```
noesis_ii/
├── core/                # 核心模块
│   ├── working_memory.py      # 工作记忆
│   ├── long_term_memory.py    # 长期记忆
│   ├── persona_profile.py     # 人格画像
│   ├── schema.py             # 数据库 Schema
├── processes/           # 后台处理进程
│   ├── consolidator.py      # 整理进程
│   ├── deepener.py         # 深化进程
│   ├── scheduler.py        # 调度器
├── retrieval/            # 检索系统
│   ├── retriever.py       # 统一检索
├── input/              # 输入模块
│   ├── input_manager.py   # 输入管理器
│   ├── book_reader.py    # 书籍阅读器
│   ├── rss_fetcher.py   # RSS 订阅器
│   └── web_scraper.py   # 网页抓取器
├── llm/                # LLM 接口
├── config_loader.py    # 配置加载器
└── main.py            # 主入口
```

---

## 6. 配置说明

配置文件：`config/default_config.yaml`

```yaml
# 种子价值观
seed_values:
  compassion: 0.8
  courage: 0.6
  integrity: 0.9
  wisdom: 0.7

# 主动阅读
active_reading:
  enabled: true
  sources: [books, rss, web]

# 时间节律
circadian_rhythm:
  evening: '22:00'
  morning: '06:00'

# LLM 配置 - OpenAI 兼容 API
llm:
  provider: openai_compatible
  api_base: https://api.longcat.chat/openai/v1
  api_key: ${LONGCAT_API_KEY}
  model: LongCat-Flash-Lite

# 存储配置
storage:
  db_path: data/noesis.db
  log_path: logs/

# 检索参数
retrieval:
  threshold: 0.5
  top_k: 10
```

### LLM 配置选项

#### 选项 1：OpenAI 兼容 API

```yaml
llm:
  provider: openai_compatible
  api_base: https://api.your-llm-provider.com/v1
  api_key: ${YOUR_API_KEY}
  model: your-model-name
```

**环境变量设置：**
```bash
# Linux/Mac
export YOUR_API_KEY="your-api-key-here"

# Windows (PowerShell)
$env:YOUR_API_KEY="your-api-key-here"
```

#### 选项 2：本地 Ollama（推荐用于隐私保护）

```yaml
llm:
  provider: ollama
  model: llama3
  base_url: http://localhost:11434
```

**Ollama 安装步骤：**

1. **安装 Ollama：**
   ```bash
   # Linux/Mac
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Windows
   # 从 https://ollama.com/download 下载安装
   ```

2. **拉取模型：**
   ```bash
   # 拉取模型（例如 Llama 3）
   ollama pull llama3
   
   # 或者使用其他模型
   ollama pull mistral
   ollama pull phi3
   ```

3. **启动 Ollama 服务：**
   ```bash
   ollama serve
   ```

#### 选项 3：OpenAI 官方 API

```yaml
llm:
  provider: openai
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o-mini
```

---

## 7. 核心模块详解

### 工作记忆 (Working Memory)

**功能**：短期快速记忆，用于即时处理记忆内容，暂存原始经验

**核心方法**：
- `capture(content, emotion, conflict_check)` - 捕获经验到工作记忆
- `get_pending(limit)` - 获取待整理的记忆
- `mark_consolidated(entry_id)` - 标记已整理
- `expire_old_entries()` - 过期条目处理

### 长期记忆 (Long-Term Memory)

**功能**：分布式长期存储记忆网络

**核心方法**：
- `create_node(content, node_type, weight, raw_anchors)` - 创建节点
- `create_link(source_node_id, target_node_id, strength, relation_type)` - 创建关联
- `retrieve(query, top_k, threshold)` - 检索
- `retrieve_with_anchors(query)` - 带事实锚点检索

### 人格画像 (Persona Profile)

**功能**：人格一致性管理，存储记忆痕迹，保持一贯性

**核心方法**：
- `store_experience(experience, ...)` - 存储经验
- `retrieve_by_conditions(conditions, top_k)` - 多条件检索
- `get_current_persona()` - 获取当前人格
- `create_trace_link(...)` - 创建痕迹关联

### 整理进程 (Consolidator)

**功能**：工作记忆 → 长期记忆和记忆痕迹的转化

**流程**：
1. 获取待整理的工作记忆
2. 使用 LLM 分析内容
3. 提取记忆线索、情绪数据
4. 提取事实锚点
5. 生成记忆痕迹
6. 建立关联

---

## 8. 使用示例

### Python API 使用

```python
from noesis_ii.main import NoesisII

# 初始化
system = NoesisII()
system.initialize('config/default_config.yaml')

# 交互模式
system.run_interactive()
```

### 工作记忆使用

```python
from noesis_ii.core.working_memory import WorkingMemory

wm = WorkingMemory('data/noesis.db')
entry_id, op = wm.capture(content)
pending = wm.get_pending()
wm.mark_consolidated(entry_id)
```

### 检索使用

```python
from noesis_ii.retrieval.retriever import Retriever

retriever = Retriever('data/noesis.db')
results = retriever.retrieve(query)
```

---

## 9. 数据库

系统使用 SQLite 数据库，存储以下核心数据：

- **working_memory**：工作记忆表
- **ltm_nodes**：长期记忆节点表
- **ltm_links**：长期记忆关联表
- **memory_traces**：记忆痕迹表

数据库初始化：
```python
from noesis_ii.core.schema import Schema
schema = Schema('data/noesis.db')
schema.create_tables()
```

---

## 10. 常见问题

### Q: 首次启动报错找不到配置文件？

系统会自动创建默认配置。若手动指定路径：
```bash
python main.py --config config/default_config.yaml
```

### Q: LLM 相关功能报错？

检查配置文件中 `llm` 段的 `api_key` 和 `api_base` 是否正确。系统支持任何 OpenAI 兼容 API。

### Q: 数据库报错或数据丢失？

数据库存储在 `data/noesis.db`。Schema 模块支持自动迁移——新增字段或表时会保留已有数据并自动升级结构。

### Q: 如何重置所有数据？

删除数据库文件即可：
```bash
del noesis_ii\data\noesis.db
```
下次启动时系统会自动重建。

---

## 许可证

本项目为个人研究项目，仅供学习与研究使用。
