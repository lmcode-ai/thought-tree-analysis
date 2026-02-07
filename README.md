# Playing Psychic: Using Thought Trees to Predict Reasoning Models Accuracy on Coding Tasks

We perform a systematic study of frontier reasoning models (e.g., DeepSeek-R1, QwQ) on real-world coding benchmarks. Our research demonstrates that the **structure** of a reasoning trace—represented as a "Thought-Tree"—is a strong predictor of model correctness, often more so than the raw content alone.

## 📌 Abstract

Recent advances in large language models (LLMs) have shown that test-time scaling can substantially improve performance on complex tasks, particularly in the coding domain. Under this paradigm, models use a larger token budget during inference to generate intermediate reasoning traces before producing a final answer. However, current evaluations primarily rely on competitive programming benchmarks, which may not capture the full range of reasoning abilities.

In this work, we:
1.  Devise a framework to **automatically generate coding tasks** of arbitrary difficulty and structure from existing benchmarks.
2.  Propose **Structured Thought-Trees** as a means to represent and analyze reasoning traces.
3.  Train a **lightweight classifier** on topological features extracted from these trees.

Our results show that this classifier can predict if a trace contains the correct answer with up to **89% accuracy** across three coding datasets.

## 🚀 Key Features

* **Task Generation Framework:** Programmatic generation of coding tasks to test specific reasoning capabilities.
* **Trace Segmentation:** NLP-based tools to break down raw LLM reasoning outputs into discrete thought segments.
* **Thought-Tree Construction:** Algorithms to reconstruct the hierarchical structure of reasoning (Continuation, Contrast, Rephrase).
* **Feature Extraction:** Extraction of graph-theoretic features (e.g., Branching Factor, Depth).
* **Correctness Classifier:** A lightweight model to predict task success based on trace structure.

## 📂 Directory Structure

```text
.
├── analysis/                   # Core logic for analyzing reasoning traces
│   ├── codelingua/            
│   ├── cruxeval/              
│   ├── safim/                  
│   ├── source_code/           
│   │   ├── clustering.py       # grouping similar reasoning patterns
│   │   ├── evaling.py          # Feature extraction (depth, branching, etc.)
│   │   ├── labeling.py         # Assigning structural labels to segments
│   │   └── segmentation.py     # NLP-based reasoning trace segmentation
│   └── tree_analysis.ipynb     # Main notebook for training the classifier
├── data_generation/            # Scripts to programmatically generate coding tasks
├── llm-evaluation/             # Raw model outputs
│   ├── codelingua/             
│   ├── cruxeval/               
│   └── safim/                 
└── README.md                   # Project documentation