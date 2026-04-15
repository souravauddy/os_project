# Intelligent Page Replacement using Classical & Reinforcement Learning Algorithms

This project implements and evaluates a wide range of **page replacement algorithms**, including both **classical OS strategies** and **reinforcement learning-based approaches**.

The objective is to analyze how different algorithms perform under realistic memory access patterns and to explore whether **learning-based methods can outperform traditional heuristics**.

---

## Algorithms Implemented

### 🔹 Classical Algorithms

- **FIFO (First-In-First-Out)**  
  - Evicts the oldest page in memory.

- **LRU (Least Recently Used)**  
  - Evicts the page that has not been used for the longest time.

- **MFU (Most Frequently Used)**  
  - Evicts the page with the highest access frequency.

- **Second Chance (Clock Algorithm)**  
  - FIFO-based method with a reference bit to give pages a second chance.

- **Optimal (OPT)**  
  - Evicts the page that will not be used for the longest time in the future (theoretical upper bound used as a performance guide for optimal page faults).

---

## 🔹 Reinforcement Learning-Based Algorithms

### 1. RL State-Based Eviction (DQN)

- Uses a Deep Q-Network to:
  - Observe cache state
  - Learn eviction policies dynamically

- State representation includes:
  - Page ID
  - Recency
  - Frequency

- Learns:
  > Which page is least useful to keep

---

### 2. Hybrid RL (LRU + MFU Selector)

- RL agent **does not directly evict pages**
- Instead, it learns:
  
  > Which policy to use: **LRU or MFU**

- At each step:
  - RL chooses between LRU / MFU
  - Selected policy performs eviction

👉 This reduces complexity and improves stability

---

# 🧠 Motivation

Traditional algorithms rely on **fixed heuristics**, which:

- Work well under specific assumptions (e.g., locality)
- Fail under changing or mixed workloads

Reinforcement Learning allows:

- Adaptive decision-making
- Learning from access patterns
- Handling non-stationary workloads

---

# ⚙️ Workload Model

We simulate realistic memory behavior using a **working set model**:

- Programs operate on a subset of pages (locality)
- Periodically shift to new working sets
- Include randomness (noise) to simulate real-world scenarios

---

# 📊 Performance Metrics

Each algorithm is evaluated using:

## 1. Hit Rate
```text
Hit Rate = Hits / Total Requests
```