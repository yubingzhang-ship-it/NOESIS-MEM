# NOESIS-II Personality-Consistent Memory System

> An intelligent assistant system with memory and self-awareness

---

## 1. System Overview

NOESIS-II is a personality-consistent memory system designed to create an intelligent assistant with memory and self-awareness. The system provides personalized and consistent outputs through multimodal inputs, hierarchical memory storage, and deep personality representation.

### Core Principles

The system employs a three-layer memory architecture:

- **Working Memory**: Short-term fast memory for immediate processing and temporary content storage
- **Long-Term Memory**: Structured long-term memory network storage
- **Memory Traces**: Deep memory traces that drive personality consistency

### Memory Processing Pipeline

1. **Capture**: Raw experience into working memory
2. **Conflict Resolution**: Deduplicate and merge similar memory content
3. **Consolidation**: Extract memory cues, emotions, and personality data using LLM
4. **Deepening**: Generate deep memory network and personality profile
5. **Retrieval**: Multi-criteria memory search
6. **Forgetting**: Soft delete and hard delete expired memories

---

## 2. Key Features

1. **Sparse Memory**: Store only LLM-uncompressible information
2. **Personality Consistency**: OCEAN Five Factor Model (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism)
3. **Memory Network**: Semantically associated memory structure
4. **Soft Delete**: Three-level forgetting mechanism (active/dormant/forgotten)
5. **Factual Anchors**: Prevent hallucinations during generative recall
6. **Multi-LLM Support**: Support for Ollama local models and remote APIs

---

## 3. Requirements & Installation

### Requirements

- Python 3.10+
- Windows / Linux / macOS

### Installation

```bash
# Navigate to project directory
cd noesis_ii

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Quick Start

```bash
# Navigate to noesis_ii directory
cd noesis_ii

# Interactive mode
python main.py

# With custom config
python main.py --config config/default_config.yaml

# Daemon mode
python main.py --mode daemon

# Execute single consolidation
python main.py --mode consolidate

# Execute deep analysis
python main.py --mode deepen
```

---

## 5. Core Architecture

```
noesis_ii/
├── core/                # Core modules
│   ├── working_memory.py      # Working memory
│   ├── long_term_memory.py    # Long-term memory
│   ├── persona_profile.py     # Persona profile
│   ├── schema.py             # Database schema
├── processes/           # Background processing
│   ├── consolidator.py      # Consolidator
│   ├── deepener.py         # Deepener
│   ├── scheduler.py        # Scheduler
├── retrieval/            # Retrieval system
│   ├── retriever.py       # Unified retriever
├── input/              # Input modules
│   ├── input_manager.py   # Input manager
│   ├── book_reader.py    # Book reader
│   ├── rss_fetcher.py   # RSS fetcher
│   └── web_scraper.py   # Web scraper
├── llm/                # LLM interfaces
├── config_loader.py    # Configuration loader
└── main.py            # Main entry point
```

---

## 6. Configuration

Config file: `config/default_config.yaml`

```yaml
# Seed values
seed_values:
  compassion: 0.8
  courage: 0.6
  integrity: 0.9
  wisdom: 0.7

# Active reading
active_reading:
  enabled: true
  sources: [books, rss, web]

# Circadian rhythm
circadian_rhythm:
  evening: '22:00'
  morning: '06:00'

# LLM configuration
llm:
  provider: openai_compatible
  api_base: https://api.longcat.chat/openai/v1
  api_key: ${LONGCAT_API_KEY}
  model: LongCat-Flash-Lite

# Storage configuration
storage:
  db_path: data/noesis.db
  log_path: logs/

# Retrieval parameters
retrieval:
  threshold: 0.5
  top_k: 10
```

---

## 7. Core Modules

### Working Memory

**Purpose**: Short-term fast memory for immediate content processing

**Key Methods**:
- `capture(content, emotion, conflict_check)` - Capture experience to working memory
- `get_pending(limit)` - Get pending memories to consolidate
- `mark_consolidated(entry_id)` - Mark as consolidated
- `expire_old_entries()` - Expire old entries

### Long-Term Memory

**Purpose**: Distributed long-term memory network storage

**Key Methods**:
- `create_node(content, node_type, weight, raw_anchors)` - Create memory node
- `create_link(source_node_id, target_node_id, strength, relation_type)` - Create association
- `retrieve(query, top_k, threshold)` - Retrieve memory
- `retrieve_with_anchors(query)` - Retrieve with factual anchors

### Persona Profile

**Purpose**: Personality consistency management, storing memory traces to maintain coherence

**Key Methods**:
- `store_experience(experience, ...)` - Store experience
- `retrieve_by_conditions(conditions, top_k)` - Multi-criteria retrieval
- `get_current_persona()` - Get current personality
- `create_trace_link(...)` - Create trace association

### Consolidator

**Purpose**: Conversion from working memory to long-term memory and memory traces

**Process**:
1. Get pending working memories
2. Analyze content using LLM
3. Extract memory cues and emotion data
4. Extract factual anchors
5. Generate memory traces
6. Create associations

---

## 8. Usage Examples

### Python API

```python
from noesis_ii.main import NoesisII

# Initialize
system = NoesisII()
system.initialize('config/default_config.yaml')

# Interactive mode
system.run_interactive()
```

### Working Memory

```python
from noesis_ii.core.working_memory import WorkingMemory

wm = WorkingMemory('data/noesis.db')
entry_id, op = wm.capture(content)
pending = wm.get_pending()
wm.mark_consolidated(entry_id)
```

### Retrieval

```python
from noesis_ii.retrieval.retriever import Retriever

retriever = Retriever('data/noesis.db')
results = retriever.retrieve(query)
```

---

## 9. Database

The system uses SQLite database to store:

- **working_memory**: Working memory table
- **ltm_nodes**: Long-term memory nodes table
- **ltm_links**: Long-term memory links table
- **memory_traces**: Memory traces table

Database initialization:
```python
from noesis_ii.core.schema import Schema
schema = Schema('data/noesis.db')
schema.create_tables()
```

---

## 10. FAQ

### Q: Config file not found on first startup?

The system automatically creates default configuration. To specify manually:
```bash
python main.py --config config/default_config.yaml
```

### Q: LLM-related errors?

Check if `api_key` and `api_base` in the `llm` section of config are correct. The system supports any OpenAI-compatible API.

### Q: Database errors or data loss?

Database is stored at `data/noesis.db`. The Schema module supports automatic migration - new fields or tables will preserve existing data and upgrade structure automatically.

### Q: How to reset all data?

Simply delete the database file:
```bash
del noesis_ii\data\noesis.db
```
The system will rebuild automatically on next startup.

---

## License

This project is for personal research and educational purposes only.
