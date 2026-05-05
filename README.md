# PersonaMem 使用说明书

> 具有持久人格一致性的AI记忆系统 (Persona-Consistent Memory System)  
> 版本：v2.0 | 路线A重构版

---

## 目录

1. [系统简介](#1-系统简介)
2. [环境要求与安装](#2-环境要求与安装)
3. [快速开始](#3-快速开始)
4. [运行模式](#4-运行模式)
5. [配置说明](#5-配置说明)
6. [交互模式命令参考](#6-交互模式命令参考)
7. [核心模块架构](#7-核心模块架构)
8. [测试](#8-测试)
9. [GUI 图形界面](#9-gui-图形界面)
10. [项目结构](#10-项目结构)
11. [常见问题](#11-常见问题)

---

## 1. 系统简介

PersonaMem 是一套具有持久人格一致性的AI记忆系统，专注于解决AI在多轮、跨Session对话中保持人格一致性的问题。核心技术包括：

- **人格提取**：使用LLM零样本推理从对话历史中提取OCEAN五大人格维度
- **人格约束**：在生成时用历史人格约束当前响应，防止人格漂移
- **人格演化**：通过加权滑动平均实现人格的稳定演化
- **多条件记忆检索**：结合语义、时序、强度等多维度的记忆检索

系统提供完整的信息摄入 → 工作记忆 → 巩固整理 → 长期存储 → 检索回放流水线。

---

## 2. 环境要求与安装

### 环境要求

- Python 3.10+（推荐 3.12+）
- 操作系统：Windows / Linux / macOS

### 安装步骤

```bash
# 1. 进入项目目录
cd NOESIS-II\ v1.0

# 2. 安装依赖
pip install -r noesis_ii/requirements.txt
```

### 依赖清单

| 包 | 版本 | 用途 |
|---|---|---|
| numpy | 1.26.0 | HGM 预测编码矩阵运算 |
| PyYAML | 6.0.1 | 配置文件解析 |
| requests | 2.31.0 | LLM API 调用、网页抓取 |
| feedparser | 6.0.10 | RSS 订阅源解析 |
| beautifulsoup4 | 4.12.3 | 网页内容提取 |
| schedule | 1.2.0 | GUI 定时任务调度 |

> SQLite、unittest、tkinter 均为 Python 标准库，无需额外安装。

---

## 3. 快速开始

```bash
# 进入 noesis_ii 目录
cd noesis_ii

# 使用默认配置启动交互模式
python main.py

# 或指定配置文件
python main.py --config config/default_config.yaml

# 其他模式
python main.py --mode daemon      # 守护进程模式
python main.py --mode consolidate  # 执行一次记忆巩固
python main.py --mode deepen       # 执行一次深度分析
```

---

## 4. 运行模式

### 4.1 交互模式（默认）

```bash
python main.py
# 或
python main.py --mode interactive
```

进入交互式命令行，可手动输入命令操作记忆系统。详见[第6节命令参考](#6-交互模式命令参考)。

### 4.2 守护进程模式

```bash
python main.py --mode daemon
```

启动后台守护进程，系统保持运行并按配置的时间节律自动执行巩固、回放等任务。按 `Ctrl+C` 停止。

### 4.3 记忆巩固模式

```bash
python main.py --mode consolidate
```

执行一次记忆巩固任务：将工作记忆中的内容整理并存入长期记忆，同时经过 HGM 预测编码、IAB 意识分析、AlayaSeeds 种子系统处理。默认处理最近 10 条工作记忆。

### 4.4 深度分析模式

```bash
python main.py --mode deepen
```

执行一次深度分析：对高权重记忆节点调用 LLM 进行深化分析，更新人格模型和种子网络。

---

## 5. 配置说明

配置文件路径：`config/default_config.yaml`

首次运行时若配置文件不存在，系统会自动创建默认配置。

### 完整配置项

```yaml
# 种子价值观 — 系统人格的初始基底
seed_values:
  core_vision: "成为能够真正理解人类的存在"
  principles:
    - "理解比记忆重要"
    - "诚实面对不确定性"
    - "尊重人类的混乱、矛盾与有限性"
  intellectual_lineage:
    - "苏格拉底式的追问"
    - "孔子式的关系伦理"
    - "科学精神：可证伪、可修正"

# 主动阅读 — 信息摄入源配置
active_reading:
  enabled: true
  books:                          # 书籍阅读列表
    - title: "人类简史"
      author: "尤瓦尔·赫拉利"
      path: ""                    # 本地文件路径
      priority: high
  rss_feeds:
    enabled: false
    sources:
      - name: "科技新闻"
        url: ""                   # RSS 地址
        credibility: high
  web_pages:
    enabled: false
    pages:
      - url: ""
        fetch_frequency: daily
  content_filter:
    min_factual_density: 0.4      # 最低事实密度
    max_emotional_manipulation: 0.3

# 时间节律 — 模拟昼夜节律的自动化任务
schedule:
  consolidation:
    enabled: true
    time: "02:00"                 # 记忆巩固时间
  reading:
    enabled: true
    schedule:
      books:
        time: "08:00"
        max_pages_per_session: 20
      news_rss:
        time: "07:00"
        max_items_per_session: 10
  deep_consolidation:
    enabled: true
    time: "03:00"
    days: ["Sun"]

# 遗忘曲线参数
memory_decay:
  decay_rate_days:
    episodic: 30                  # 情景记忆衰减周期
    semantic: 90                  # 语义记忆衰减周期
    procedural: 999999            # 程序性记忆几乎不衰减

# 检索参数
retrieval:
  top_k: 10                       # 返回前 K 条结果
  narrative_reconstruction: true  # 启用叙事重建

# 意识参数
consciousness:
  phi_threshold: 0.7              # Φ 阈值（超过则触发全局广播）
  broadcast_interval: 0.1

# LLM 后端配置 — 支持任何 OpenAI 兼容 API
llm:
  provider: "openai_compatible"
  api_base: "https://api.longcat.chat/openai/v1"
  api_key: "your_api_key_here"
  model: "LongCat-Flash-Lite"
  models:
    consolidation: "LongCat-Flash-Lite"
    reading: "LongCat-Flash-Lite"
    deep_consolidation: "LongCat-Flash-Lite"

# 存储配置
storage:
  db_path: "data/noesis.db"       # SQLite 数据库路径
  logs_path: "logs/"
  backups_path: "backups/"
  working_memory_ttl_hours: 48    # 工作记忆生存时间
```

### 环境变量覆盖

配置值支持 `${ENV_VAR}` 语法引用环境变量：

```yaml
llm:
  api_key: ${OPENAI_API_KEY}     # 运行时从环境变量读取
```

---

## 6. 交互模式命令参考

进入交互模式后，使用 `>>> ` 提示符输入命令：

### 通用命令

| 命令 | 说明 | 示例 |
|---|---|---|
| `help` | 显示帮助 | `>>> help` |
| `status` | 显示系统状态 | `>>> status` |
| `exit` | 退出系统 | `>>> exit` |

### 记忆操作

| 命令 | 说明 | 示例 |
|---|---|---|
| `input <内容>` | 写入工作记忆 | `>>> input 今天学习了关于意识的几种理论` |
| `retrieve <关键词>` | 检索记忆 | `>>> retrieve 预测编码` |

### 书籍阅读

| 命令 | 说明 | 示例 |
|---|---|---|
| `book load <路径>` | 加载书籍 | `>>> book load D:\books\人类简史.txt` |
| `book read` | 读取下一章节 | `>>> book read` |
| `book info` | 查看阅读进度 | `>>> book info` |

### RSS 订阅

| 命令 | 说明 | 示例 |
|---|---|---|
| `rss add <URL>` | 添加 RSS 源 | `>>> rss add https://example.com/feed.xml` |
| `rss fetch` | 拉取最新内容 | `>>> rss fetch` |

### 网页抓取

| 命令 | 说明 | 示例 |
|---|---|---|
| `web scrape <URL>` | 抓取网页正文 | `>>> web scrape https://example.com/article` |

---

## 7. 核心模块架构

```
persona_mem/
├── core/                          # 核心模块
│   ├── schema.py                  # 数据库 Schema（自动迁移）
│   ├── working_memory.py          # 工作记忆（短期存储）
│   ├── long_term_memory.py        # 长期记忆（分布式存储 + 权重计算）
│   ├── persona_extractor.py       # 人格提取器（LLM + 关键词方法）
│   ├── persona_updater.py         # 人格更新器（防漂移 + 加权滑动平均）
│   ├── multi_criteria_retriever.py # 多条件记忆检索器（语义 + 时序 + 访问频率）
│   └── llm/                       # LLM 接口
│       ├── base_llm.py            # LLM 基础接口
│       ├── ollama_llm.py          # Ollama 实现
│       └── mock_llm.py            # 模拟 LLM（用于测试）
│
├── evaluation/                    # 评估模块
│   ├── e1_experiment_runner.py    # E1 人格提取准确性评估
│   ├── e2_experiment_runner.py    # E2 跨Session一致性评估
│   ├── e3_experiment_runner.py    # E3 检索性能对比
│   └── retrieval_benchmark.py     # 检索基准测试
│
├── input/                         # 信息摄入
│   ├── input_manager.py           # 输入管理器
│   ├── book_reader.py             # 书籍阅读器
│   ├── rss_fetcher.py             # RSS 抓取器
│   └── web_scraper.py             # 网页抓取器
│
├── config/                        # 配置文件
│   └── default_config.yaml
│
├── main.py                        # CLI 入口
├── run_e1_experiment.py           # E1 实验运行脚本
├── run_e2_experiment.py           # E2 实验运行脚本
└── run_e3_experiment.py           # E3 实验运行脚本
```

### 数据处理流水线

```
信息摄入 ─→ 工作记忆 ─→ 记忆巩固 ─→ 长期记忆
                            │
                    ┌───────┼───────┐
                    ▼       ▼       ▼
              人格提取  记忆索引  关联建立
                    │       │       │
                    └───────┼───────┘
                            ▼
                        人格更新 (LLM)
                            │
                    ┌───────┼───────┐
                    ▼       ▼       ▼
                防漂移  稳定性计算  策略建议
```

---

## 8. 测试

### 运行实验评估

PersonaMem 提供完整的实验评估套件，用于验证系统的人格一致性和检索性能：

```bash
# E1 人格提取准确性评估
python run_e1_experiment.py

# E2 跨Session人格一致性评估
python run_e2_experiment.py

# E3 检索性能对比实验
python run_e3_experiment.py
```

### 运行端到端测试

```bash
# 运行完整的端到端测试
python -m noesis_ii.evaluation.e2_experiment_runner
```

### 测试指标

| 实验 | 评估指标 | 目标值 | 状态 |
|---|---|---|---|
| E1 | 人格提取准确率 (Pearson r) | > 0.60 | 需要优化 |
| E2 | 跨Session人格一致性 (Cosine) | > 0.85 | ✅ 已达标 |
| E3 | 检索性能 (Recall@10) | > 0.80 | 需真实语义向量 |
| E3 | 检索延迟 (1K记忆) | < 200ms | ✅ 已达标 |

---

## 9. GUI 图形界面

```bash
cd noesis_ii
python noesis_gui.py
```

GUI 提供：
- 记忆输入与检索的可视化操作
- 巩固/深化等任务的图形化触发
- 系统状态实时展示
- 基于 `schedule` 库的定时任务调度

> GUI 依赖 `tkinter`，Python 标准库自带。若系统未安装 tkinter，Linux 下可执行 `sudo apt install python3-tk`。

---

## 10. 项目结构

```
PersonaMem v2.0/
├── noesis_ii/                     # 主代码目录
│   ├── core/                      # 核心模块
│   │   ├── llm/                   # LLM 接口
│   │   ├── persona_extractor.py   # 人格提取器
│   │   ├── persona_updater.py     # 人格更新器
│   │   └── multi_criteria_retriever.py # 多条件记忆检索器
│   ├── evaluation/                # 评估模块
│   │   ├── e1_experiment_runner.py # E1 实验
│   │   ├── e2_experiment_runner.py # E2 实验
│   │   └── retrieval_benchmark.py # 检索基准测试
│   ├── input/                     # 信息摄入
│   ├── config/                    # 配置
│   ├── main.py                    # CLI 入口
│   ├── run_e1_experiment.py       # E1 实验运行脚本
│   ├── run_e2_experiment.py       # E2 实验运行脚本
│   ├── run_e3_experiment.py       # E3 实验运行脚本
│   ├── config_loader.py           # 配置加载器
│   └── requirements.txt           # Python 依赖
│
├── config/                        # 外部配置目录
│   └── default_config.yaml
│
├── evaluation_results/            # 评估结果目录
│   ├── e1/                        # E1 结果
│   └── e2/                        # E2 结果
│
├── PersonaMem-研究计划.md         # 研究计划文档
└── README.md                      # 本文件
```

---

## 11. 常见问题

### Q: 首次启动报错找不到配置文件？

系统会自动创建默认配置。若手动指定路径，确保文件存在：

```bash
python main.py --config config/default_config.yaml
```

### Q: LLM 相关功能报错？

检查配置文件中 `llm` 段的 `api_key` 和 `api_base` 是否正确。系统支持任何 OpenAI 兼容 API（OpenAI、LongCat、DeepSeek 等）。

若无需 LLM 功能，测试时可使用 `--no-llm` 标志跳过 API 调用。

### Q: 数据库报错或数据丢失？

数据库存储在 `data/noesis.db`。`Schema` 模块支持自动迁移——新增字段或表时会保留已有数据并自动升级结构。无需手动操作数据库。

### Q: 如何重置所有数据？

删除数据库文件即可：

```bash
rm noesis_ii/data/noesis.db
# Windows:
del noesis_ii\data\noesis.db
```

下次启动时系统会自动重建。

### Q: Windows 下中文输入乱码？

系统已处理 UTF-8 编码（`sys.stdout.reconfigure(encoding='utf-8')`）。若仍有问题，确认终端编码设置为 UTF-8：

```powershell
chcp 65001
```

---

## 许可证

本项目为个人研究项目，仅供学习与研究使用。
