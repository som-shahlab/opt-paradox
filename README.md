# 🩺 Clinical Diagnosis Agent

*A framework for testing clinical reasoning capabilities of language model agents using LangGraph.*

---

## 🧠 Overview

This project implements single-agent and multi-agent workflows to evaluate the end-to-end clinical diagnosis capabilities of language models (LLMs). The system:

* Reads patient histories
* Requests and interprets physical exams, lab tests, and imaging
* Provides final diagnoses and treatment recommendations
* Evaluates diagnostic accuracy and efficiency

Supported LLM backends include Azure OpenAI, Anthropic Claude, Google Gemini, Meta Llama, OpenAI o3-mini, and DeepSeek.

---

## 🦠 Supported Pathologies

* Appendicitis
* Cholecystitis
* Pancreatitis
* Diverticulitis

---

## 📝 Main Findings

*(Main findings will be updated here upon paper publication.)*

<details>
  <summary>
  	<b>Main Findings</b>
  </summary>

*(Details will be provided here.)*

</details>

---

## 🗓️ Updates

* *(Example)* June 2025 : Our preprint is available online!

---

## 📈 Reproducing Results

**Note:** To replicate results, copy the template and configure your own settings:

```bash
cp config.example.yaml config.yaml
```

Then edit `config.yaml` with your actual API keys, paths, and endpoint URLs.

All code was tested with Python v3.10. Cloud-hosted models (Azure, Claude, Gemini) run on standard personal computers.

### ⚙️ Installation

```bash
conda env create -f environment.yaml
conda activate clinagent_env
pip install -r requirements.txt
```

---

### 🧪 Single-Agent Pipeline

```bash
python run_single_agent.py \
  --model_id_main <platform> \
  --model_id_matcher <platform> \
  --dataset_type {train,val,test} \
  [--log_to_file] \
  [--log_filename LOGFILE] \
  [--concurrency N]
```

*Example:*

```bash
python run_single_agent.py \
  --model_id_main gpt \
  --model_id_matcher claude \
  --dataset_type val \
```

---

### 🧬 Multi-Agent Pipeline

```bash
python run_multi_agent.py \
  --model_id_info <platform> \
  --model_id_interpretation <platform> \
  --model_id_matcher <platform> \
  --model_id_diagnosis <platform> \
  --dataset_type {train,val,test} \
  [--log_to_file] \
  [--log_filename LOGFILE] \
  [--concurrency N]
```

*Example:*

```bash
python run_multi_agent.py \
  --model_id_info gemini \
  --model_id_interpretation claude \
  --model_id_matcher llama \
  --model_id_diagnosis gpt \
  --dataset_type test \
  --log_to_file \
```

---

### 📊 Evaluating Logs

After running either pipeline, evaluate the generated logs using:

```bash
python run_evals.py --log_dir logs/<log_directory>
```

*Example:*

```bash
python run_evals.py --log_dir logs/gpt_gpt_val
```

Results are saved in the `results/` directory.

---

## 📚 Citation

*(Placeholder for future publication citation.)*

```bibtex
@article{your_citation_here,
  title={The Optimization Paradox: When Best Components Yield Suboptimal Clinical AI Systems},
  author={Your Name and others},
  journal={Journal Name},
  year={202X}
}
```

---

## 📧 Issues

Please report issues by creating an issue on this GitHub repository.

---
