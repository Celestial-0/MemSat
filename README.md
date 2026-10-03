# MemSat

Research project on how much long-term memory an LLM agent actually needs when answering queries.

## Research Title

> **Memory Retrieval Saturation in LLM Agents: An Empirical Study of Accuracy and Efficiency**

## Research Area

**AI/ML**

## Research Idea

The project investigates **how much long-term memory an LLM agent actually needs when answering queries**.

The core idea is to systematically vary the number of retrieved memories while keeping the rest of the experimental setup fixed. The approved form specifies retrieval depths:

```text
K = 1, 2, 4, 8, 16, 32, 64
```

The experiment measures how retrieval depth affects:

| Metric | What it measures |
|---|---|
| **Answer Accuracy** | Quality/correctness of the agent's answer |
| **Relevant-Memory Coverage** | How much useful information is retrieved |
| **Token Usage** | Additional context consumed |
| **Latency** | Retrieval/inference response time |
| **Inference Cost** | Computational cost of larger contexts |

The central research question is:

> **Does increasing the number of retrieved memories continuously improve an LLM agent's performance, or does retrieval eventually reach a saturation point where additional memories provide only marginal or negligible benefit?**

## Proposed Experimental Design

The approved abstract keeps the following components **fixed**:

- Long-term memory system
- Memory representation
- Embedding model
- Retrieval method
- Language model
- Prompts
- Dataset
- Evaluation procedure

Only **retrieval depth `K`** changes.

This makes the study essentially an empirical **retrieval-depth ablation study**.

### Experimental Flow

```text
LLM Agent Query
      │
      ▼
Long-Term Memory
      │
      ▼
Retrieve K memories
(K = 1,2,4,8,16,32,64)
      │
      ▼
Generate Answer
      │
      ▼
Evaluate
 ┌───────────────┐
 │ Accuracy      │
 │ Memory Cover  │
 │ Tokens        │
 │ Latency       │
 │ Cost          │
 └───────────────┘
      │
      ▼
Identify Saturation Point
```

## Research Hypothesis / Core Idea

The expected relationship is:

```text
Retrieval Depth ↑
       │
       ├── Accuracy ↑ initially
       │
       ├── Relevant-memory coverage ↑
       │
       ├── Token usage ↑
       ├── Latency ↑
       └── Cost ↑
                │
                ▼
       Saturation Region
                │
                ▼
Additional memories
→ marginal / negligible accuracy gains
```

The project also proposes investigating whether the saturation point **differs across different memory-reasoning task types**.

## Main Research Contribution

The intended contribution is an empirical characterization of the **accuracy-efficiency trade-off in long-term memory retrieval for LLM agents**.

In practical terms, the research aims to answer:

> **What is the optimal retrieval depth before additional memory becomes inefficient?**

The results can then provide guidance for designing LLM-agent memory systems and selecting an appropriate retrieval depth.

### One-line Research Idea

> **Measure where long-term memory retrieval in LLM agents stops producing meaningful accuracy gains, and quantify the token, latency, and computational cost of retrieving beyond that point.**