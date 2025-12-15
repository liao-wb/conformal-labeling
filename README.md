# Conformal Labeling: Selective Labeling with False Discovery Rate Control

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official implementation of the paper **"Selective Labeling with False Discovery Rate Control"** (Under Review at ICLR 2026).

## Abstract

Obtaining high-quality labels for large datasets is expensive. While AI models offer a cost-effective alternative, their inherent labeling errors compromise data quality. Existing methods lack theoretical guarantees on the quality of AI-assigned labels.

We introduce **Conformal Labeling**, a rigorous method to identify subsets where AI predictions can be provably trusted. By formulating selective labeling as a multiple-hypothesis testing problem and leveraging conformal p-values, our method strictly controls the **False Discovery Rate (FDR)**—the expected proportion of incorrect labels in the selected subset—below a user-specified level (e.g., 10%).

## Project Structure

```text
.
├── algorithm/               # Core algorithm implementation (Conformal Labeling)
│   ├── select_alg.py        # Main selection logic (BH, Storey-BH, etc.)
│   └── preprocess.py        # Data preprocessing utilities
├── image_classification/    # ImageNet experiments (ResNet, CLIP)
├── llm/                     # LLM QA experiments (MMLU, MedMCQA)
├── rlhf_labeling/           # RLHF preference labeling (HH-RLHF, TLDR)
├── multimodel/              # Sequential selection experiments
├── regression/              # Regression tasks (AlphaFold)
├── selective_prediction/    # Baseline comparisons (Risk Control / SGR)
├── plot_utils/              # Visualization scripts
├── datasets/                # Placeholder for dataset CSVs
└── requirements.txt         # Dependencies