<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=220&section=header&text=FineTuned-Qwen7B-Model&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Domain-Adapted%20Astronomy%20%26%20Exoplanet%20Reasoning%20LLM&descAlignY=55&descSize=18" />

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&duration=3000&pause=800&color=58A6FF&center=true&vCenter=true&width=780&lines=Qwen2.5-7B-Instruct+%E2%86%92+Astronomy+Reasoning+Specialist;QLoRA+%2B+4-bit+NF4+Quantization+on+Kaggle+T4+x2;40.37M+%2F+7.66B+Parameters+Trained+(0.53%25);Trained+on+32%2C483+Real+NASA+Exoplanet+Archive+Records" alt="Typing SVG" />

<br/>

[![Model](https://img.shields.io/badge/Base_Model-Qwen2.5--7B--Instruct-blueviolet?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
[![Method](https://img.shields.io/badge/Method-QLoRA_4--bit_NF4-orange?style=for-the-badge&logo=lightning&logoColor=white)](#-fine-tuning-method)
[![Params](https://img.shields.io/badge/Trainable_Params-40.37M_(0.53%25)-2ea44f?style=for-the-badge)](#-trainable-parameters)
[![Domain](https://img.shields.io/badge/Domain-Astronomy_%7C_Exoplanets-1f6feb?style=for-the-badge&logo=nasa&logoColor=white)](#-dataset)

[![GitHub Repo stars](https://img.shields.io/github/stars/aashutoshkumarbhardwaj/FineTuned-Qwen7B-Model?style=social)](https://github.com/aashutoshkumarbhardwaj/FineTuned-Qwen7B-Model)
[![License](https://img.shields.io/badge/License-Apache_2.0-yellow.svg?style=flat-square)](#-license)
[![Made with PyTorch](https://img.shields.io/badge/PyTorch-2.10.0+cu128-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](#)
[![PEFT](https://img.shields.io/badge/PEFT-0.20.0-4B8BBE?style=flat-square)](#)

</div>

<br/>

## 🪐 Overview

**FineTuned-Qwen7B-Model** is a domain-adaptation project that takes `Qwen/Qwen2.5-7B-Instruct` and specializes it for **astronomy and exoplanet reasoning** — orbital mechanics, planetary property interpretation, discovery-method explanation, and stellar dynamics — using real observational data from the **NASA Exoplanet Archive**.

This is **not** a full 7B retrain. Using **QLoRA** with 4-bit NF4 quantization, only lightweight adapter layers were trained — **40.37M parameters (0.53%)** of the 7.66B-parameter base model — while the frozen base stayed in 4-bit precision.

> **In one sentence:** A domain-adapted Qwen2.5-7B astronomy model trained with QLoRA on 15,845 instruction-response examples generated from real NASA exoplanet observations, with only 0.53% of model parameters updated.

<br/>

## 🚀 Pipeline

```text
NASA Exoplanet Archive
        │
   32,483 raw observations
        │
   Data cleaning + filtering
        │
   Task-specific reasoning examples
        │
   15,845 generated examples
        │
   Train / Validation / Test split
        │
   Qwen2.5-7B-Instruct baseline
        │
   4-bit NF4 quantization
        │
   QLoRA (rank 16, alpha 32)
        │
   40.37M trainable parameters
        │
   3,178 training steps · 2 epochs
        │
   Best checkpoint (checkpoint-1589)
        │
   154 MB LoRA adapter
        │
   Evaluation (numerical + qualitative)
        │
   Kaggle + Hugging Face release
```

<br/>

## 📊 Dataset

Built from **32,483 raw exoplanet records** covering planet name, host star, discovery method/year, orbital period, planet radius/mass, equilibrium temperature, stellar temperature/mass/radius, and system distance. After cleaning and filtering, **15,845 instruction-response examples** were generated across four reasoning tasks.

### Split

| Split | Examples |
|---|---:|
| Train | 12,711 |
| Validation | 1,552 |
| Test | 1,582 |
| **Total** | **15,845** |

### Task distribution

| Task | Train | Validation | Test |
|---|---:|---:|---:|
| Observation interpretation | 4,725 | 571 | 581 |
| Orbital reasoning | 4,725 | 571 | 581 |
| Planet property reasoning | 863 | 102 | 126 |
| Stellar dynamics | 2,398 | 308 | 294 |

<br/>

## 🧠 Base Model

| | |
|---|---|
| **Model** | `Qwen/Qwen2.5-7B-Instruct` |
| **Parameters** | ~7.66 billion |
| **Vocabulary** | 151,643 tokens |
| **Chat template** | Native Qwen2.5 chat template used for all training examples |

<br/>

## ⚙️ Fine-Tuning Method

**QLoRA** — 4-bit NF4 quantization, double quantization, FP16 compute, LoRA adapters on both attention and MLP projections.

### LoRA configuration

| Parameter | Value |
|---|---:|
| Rank | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Bias | none |
| Task type | Causal LM |

**Target modules:** `q_proj` `k_proj` `v_proj` `o_proj` `gate_proj` `up_proj` `down_proj`

<br/>

## 🔢 Trainable Parameters

<div align="center">

| Metric | Value |
|---|---:|
| Total parameters | 7,655,986,688 |
| Trainable parameters | 40,370,176 |
| **Trainable %** | **0.5273%** |

</div>

Only **~0.53%** of the model was updated — the rest stayed frozen in 4-bit precision. This is the core efficiency argument of parameter-efficient fine-tuning.

<br/>

## ☁️ Hardware & Environment

Trained on **Kaggle** with dual Tesla T4 GPUs.

```text
GPU 0: Tesla T4          GPU memory: 15,360 MiB
GPU 1: Tesla T4          GPU memory: 15,360 MiB
CUDA:              True
PyTorch:           2.10.0+cu128
Transformers:      5.16.1
Datasets:          5.0.1
PEFT:              0.20.0
TRL:               1.12.0
bitsandbytes:      0.50.2
```

<br/>

## 🏋️ Training Configuration

| Setting | Value |
|---|---:|
| Epochs | 2 |
| Training examples | 12,711 |
| Optimization steps | 3,178 |
| Batch size / device | 1 |
| Gradient accumulation | 8 |
| Effective batch size | 8 |
| Learning rate | 2e-4 |
| Optimizer | paged_adamw_8bit |
| FP16 | ✅ |
| Gradient checkpointing | ✅ |
| **Training time** | **5h 52m 49s** (~21,176s) |

<br/>

## 📉 Training Results

| Epoch | Training Loss | Validation Loss |
|---|---:|---:|
| 1 | 0.213586 | **0.233372** ✅ best |
| 2 | 0.201184 | 0.236410 |

**Final training loss:** 0.216277 · **Best checkpoint:** `checkpoint-1589` · **Best validation loss:** `0.233371526`

> ⚠️ **Note:** Epoch 2 lowered training loss but slightly worsened validation loss — the best checkpoint came from epoch 1, not the final epoch. Checkpoint selection was based on validation loss, not just "last step."

<br/>

## 🧪 Before vs. After — Numerical Reasoning

**Example 1**
```text
Input:   Star mass = 0.330 solar masses, Orbital period = 9.715 days
Reference:        0.062 AU
Baseline (before): failed to complete the numerical calculation
Fine-tuned (after): 0.060 AU   →  ~3.23% relative error
```

**Example 2 — Kepler-328 c**
```text
Input:   Star mass = 1.150 solar masses, Orbital period = 71.312 days
Reference:        0.353 AU
Fine-tuned:       0.364 AU     →  ~3.12% relative error
```

<br/>

## 🧠 Qualitative Reasoning Samples

<details>
<summary><strong>Stellar dynamics</strong></summary>
<br/>

**Q:** What can a 1.080 solar-mass host star tell us about the orbital dynamics?
**Model:** The host star has a mass broadly comparable to the Sun, so the orbital dynamics are on a roughly solar-like scale.
</details>

<details>
<summary><strong>Radial velocity detection</strong></summary>
<br/>

**Q:** What does the radial velocity discovery method tell us?
**Model:** Correctly explained that the planet is detected through changes in the host star's measured line-of-sight velocity.
</details>

<details>
<summary><strong>Transit detection</strong></summary>
<br/>

**Model:** Correctly explained that a planet is detected when it passes in front of its host star and causes a measurable decrease in observed brightness.
</details>

<br/>

## 📦 Released Artifact

The release is a **LoRA adapter**, not a standalone 7B model — the base model is loaded separately at inference time.

| File | Size |
|---|---:|
| `adapter_model.safetensors` | ~154.05 MB |
| `tokenizer.json` | ~10.89 MB |
| `adapter_config.json` | ~1 KB |
| `tokenizer_config.json` | ~1 KB |
| `chat_template.jinja` | ~2 KB |
| `training_args.bin` | ~5 KB |
| `README.md` | — |

<br/>

## 🏆 Key Numbers

<div align="center">

| Metric | Value |
|---|---:|
| Raw astronomical observations | 32,483 |
| Generated instruction-response examples | 15,845 |
| Training examples | 12,711 |
| Reasoning task categories | 4 |
| Base model parameters | 7.66B |
| Trainable parameters | 40.37M |
| Parameters updated | 0.53% |
| Quantization | 4-bit NF4 |
| Optimization steps | 3,178 |
| Training epochs | 2 |
| Best validation loss | 0.2334 |
| Final adapter size | 154 MB |
| Orbital-distance relative error | ~3.2% |

</div>

<br/>

## 🆚 Why This Project Is Interesting

Not just *"I fine-tuned Qwen."* The full story:

> Built a domain-specific reasoning dataset from real astronomical observations → established a baseline → adapted a 7B instruction model using QLoRA on commodity Kaggle GPUs → measured validation loss and numerical error → selected the best checkpoint by validation performance, not just final step → released the resulting adapter.

<br/>

## ⚠️ Scope & Honest Limitations

- Reported error (~3.2%) is demonstrated on **representative orbital-distance examples**, not an aggregate test-set metric across all 1,582 test examples.
- No rigorous aggregate baseline-vs-fine-tuned benchmark was run across the full test set — improvements are shown via loss curves plus targeted qualitative/numerical examples.
- This adapter is a **research/portfolio artifact**, not a production-grade scientific tool — outputs should not be used for real astronomical calculations without independent verification.

<br/>

## 🛠️ Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

model = PeftModel.from_pretrained(base_model, "<your-hf-username>/FineTuned-Qwen7B-Model")

messages = [{"role": "user", "content": "A planet orbits a 0.9 solar-mass star with a period of 45 days. Estimate the orbital distance in AU."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

output = model.generate(inputs, max_new_tokens=256, temperature=0.3)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

<br/>

## 📁 Repository Structure

```text
FineTuned-Qwen7B-Model/
├── notebooks/
│   ├── finetune_v1.ipynb
│   ├── finetune_v2.ipynb
│   └── finetune_v3.ipynb
├── adapter/
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── chat_template.jinja
├── data/
│   └── (dataset generation / cleaning scripts)
└── README.md
```

<br/>

## 📄 License

Released under the **Apache 2.0** License. Base model weights remain subject to the original Qwen2.5 license terms.

<br/>

## 🙌 Acknowledgements

- **Qwen Team** — for `Qwen2.5-7B-Instruct`
- **NASA Exoplanet Archive** — for the observational dataset
- **Kaggle** — for free dual-T4 GPU compute

<br/>

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2c5364,50:203a43,100:0f2027&height=100&section=footer" />

**If this project is useful to you, consider ⭐ starring the repo.**

</div>
