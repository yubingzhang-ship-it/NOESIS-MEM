
# NOESIS-II Design and Implementation Document

## 1. System Overview

NOESIS-II is a personality-consistent memory system designed to create an intelligent assistant with memory and self-awareness. The system provides personalized and consistent outputs through multimodal inputs, hierarchical memory storage, and deep personality representation.

## 2. Core Principles

### 2.1 Memory System Principles

NOESIS-II employs a three-layer memory architecture:

- **Working Memory**: Short-term fast memory for immediate processing and temporary content storage
- **Long-Term Memory**: Structured long-term memory network storage
- **Memory Traces**: Deep memory traces that drive personality consistency


### 2.2 Memory Processing Pipeline

The system processes memory experiences in the following pipeline:

1. **Capture**: Raw experience into working memory
2. **Conflict Resolution**: Deduplicate and merge similar memory content
3. **Consolidation**: Extract memory cues, emotions, and personality data using LLM
4. **Deepening**: Generate deep memory network and personality profile
5. **Retrieval**: Multi-criteria memory search
6. **Forgetting**: Soft delete and hard delete expired memories

### 2.3 Personality Consistency Principles

Maintaining response consistency through memory traces and the OCEAN Five Factor Model:

- **Openness**: Curiosity, creativity
- **Conscientiousness**: Self-discipline, organization
- **Extraversion**: Sociability, energy
- **Agreeableness**: Cooperation, trust
- **Neuroticism**: Emotional stability (low = stable)

## 3. Core Architecture

```
noesis_ii/
├── core/                # Core modules
├── processes/           # Background processing
├── retrieval/          # Retrieval layer
├── input/            # Input layer
├── llm/              # LLM interfaces
├── config_loader.py   # Configuration loader
└── main.py           # Main entry point
```

## 4. Core Module Details

### 4.1 Working Memory
**Purpose**: Short-term fast memory for immediate content processing

**Key Methods**:
- `capture(content, emotion, conflict_check)` Capture experience to working memory
- `get_pending(limit)` Get pending memories to consolidate
- `mark_consolidated(entry_id)` Mark as consolidated
- `expire_old_entries()` Expire old entries
- `get_hot_context(n, format)` Get hot cache
- `soft_forget()` Soft delete

**Table Design**:
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
  emotion_data  # Structured emotion JSON
)
```

### 4.2 Long-Term Memory
**Purpose**: Distributed long-term memory network storage

**Key Methods**:
- `create_node(content, node_type, weight, raw_anchors)` Create memory node
- `create_link(source_node_id, target_node_id, strength, relation_type)` Create association
- `retrieve(query, top_k, threshold)` Retrieve memory
- `access_node(node_id)` Update access time
- `apply_forgetting()` Apply forgetting
- `retrieve_with_anchors(query)` Retrieve with factual anchors

**Table Structure**:
```sql
ltm_nodes(
  id,
  content,
  type,
  weight,
  created_at,
  last_accessed,
  raw_anchors  # Factual anchors JSON
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

### 4.3 Persona Profile
**Purpose**: Personality consistency management, storing memory traces to maintain coherence

**Core Data Structures**:
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

**Key Methods**:
- `store_experience(experience, ...)` Store experience
- `retrieve_by_conditions(conditions, top_k)` Multi-criteria retrieval
- `get_current_persona()` Get current personality
- `soft_forget()` Soft delete
- `create_trace_link(...)` Create trace association
- `recover_trace(db_id)` Recover forgotten memory

**Table Structure**:
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

### 4.4 Hierarchical Generative Model (HGM)
**Purpose**: Predictive coding and free energy principle

### 4.5 Extended Mind
**Purpose**: External resource management and structural coupling

## 5. Background Processing Modules

### 5.1 Consolidator
**Purpose**: Conversion from working memory to long-term memory and memory traces

**Process**:
1. Get pending working memories
2. Analyze content using LLM
3. Extract memory cues and emotion data
4. Extract factual anchors
5. Generate memory traces
6. Create associations

**Key Methods**:
- `run(limit, batch_size, max_workers, async_mode)` Run consolidation
- `_consolidate_entry(entry)` Consolidate single entry
- `_analyze_with_llm(content)` LLM analysis
- `_analyze_persona(content, emotion)` Personality analysis
- `_extract_anchors(raw_content)` Extract factual anchors
- `build_persona_prompt()` Build personality prompt

### 5.2 Deepener
**Purpose**: Deep memory network construction, deep personality generation

### 5.3 Replay Engine
**Purpose**: Memory replay

### 5.4 Scheduler
**Purpose**: Background process scheduling

## 6. Retrieval System

### 6.1 Unified Retriever
**Purpose**: Multi-source memory unified retrieval

### 6.2 Hybrid Retriever
**Purpose**: Semantic + structural hybrid retrieval

## 7. Input Modules

### 7.1 Input Manager
**Purpose**: Manage multimodal inputs

### 7.2 Active Inputs
- **Book Reader**
- **RSS Fetcher**
- **Web Scraper**

## 8. LLM Interfaces

### 8.1 LongCat Client
**Purpose**: LongCat API calls

### 8.2 Ollama LLM
**Purpose**: Local Ollama model calls

## 9. Configuration Management

**Configuration File**: config/default_config.yaml

**Configuration Items**:
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

## 10. Usage Guide

### 10.1 Main Entry Usage

**Initialization**:
```python
from noesis_ii.main import NoesisII
system = NoesisII()
system.initialize(config_path)
```

**Interactive Mode**:
```bash
python main.py --mode interactive
```

**Consolidation Mode**:
```bash
python main.py --mode consolidate
```

### 10.2 Working Memory Usage

```python
from noesis_ii.core.working_memory import WorkingMemory
wm = WorkingMemory(db_path)
entry_id, op = wm.capture(content)
pending = wm.get_pending()
wm.mark_consolidated(entry_id)
```

### 10.3 Retrieval Usage

```python
from noesis_ii.retrieval.retriever import Retriever
retriever = Retriever(db_path)
results = retriever.retrieve(query)
```

## 11. Database Initialization and Migration

**Schema Initialization**:
```python
from noesis_ii.core.schema import Schema
schema = Schema(db_path)
schema.create_tables()
```

## 12. Key Features

1. **Sparse Memory**: Store only LLM-uncompressible information
2. **Personality Consistency**: OCEAN Five Factor Model
3. **Memory Network**: Semantically associated memory structure
4. **Soft Delete**: Three-level forgetting (active/dormant/forgotten)
5. **Factual Anchors**: Prevent hallucinations during generative recall
6. **Local/Remote LLM**: Support for both Ollama and remote APIs

