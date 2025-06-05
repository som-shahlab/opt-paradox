# 🔬 Optimization Paradox in Multi-Agent Systems

This repo contains code for the **"The Optimization Paradox in Clinical AI Multi-Agent Systems"** paper. It demonstrates how **optimizing individual components can catastrophically undermine overall system performance** in multi-agent clinical AI systems. The framework enables **evaluation of both single-agent and multi-agent workflows** on real patient cases from the MIMIC-CDM dataset using multiple LLM families.

It currently supports **8 different LLM families** and provides comprehensive evaluation metrics including diagnostic accuracy, process adherence, and cost efficiency.

**📖 Table of Contents**
1. [🚀 Quick Start](#-quick-start)
2. [📊 What This Does](#-what-this-does)
3. [🏥 Key Finding](#-key-finding)
4. [📈 Results & Evaluation](#-results--evaluation)
5. [🔧 Supported Models](#-supported-models)
6. [📋 Requirements](#-requirements)
7. [📚 Citation](#-citation)
8. [📧 Issues](#-issues)

---

## 🚀 Quick Start

1. **Install dependencies**
  ```bash
  conda env create -f environment.yaml
  conda activate clinagent_env
  ```

2. **Configure APIs**
  ```bash
  cp config.example.yaml config.yaml
# Edit config.yaml with your API keys
  ```

3. **Run evaluation**
  ```bash
  # Single agent
  python3 run_single_agent.py --model_id_main gpt --dataset_type val

  # Multi-agent 
  python3 run_multi_agent.py --model_id_info gemini --model_id_diagnosis gpt --dataset_type val
  ```

## 📊 What This Does
Tests clinical reasoning on **2,400 real patient cases** across **4 abdominal conditions**:

- **Single-agent:** One model handles everything  
- **Multi-agent:** Specialized models for information gathering, interpretation, and diagnosis  
- **Best-of-Breed:** Top-performing components combined _(spoiler: performs worst!)_

## 🏥 Key Finding
The *Best-of-Breed* system built from individually optimal components achieved only 67.7% accuracy vs 77.4% for a well-integrated multi-agent system, despite superior process metrics.

## 📈 Results & Evaluation
  ```bash
  python3 run_evals.py --log_dir logs/<experiment_name>
  ```
Results include diagnostic accuracy, process adherence, and cost metrics.

## 🔧 Supported Models
Azure OpenAI, Claude, Gemini, Llama, o3-mini, DeepSeek

## 📋 Requirements

- Python 3.10+  
- API keys for your chosen models  
- [MIMIC-CDM dataset access](https://physionet.org/content/mimic-iv-ext-cdm/1.1/) 

## 📚 Citation

*(Placeholder for future publication citation.)*

---

## 📧 Issues

Please report issues by creating an issue on this GitHub repository.

---
