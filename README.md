# GI-Classification

A machine learning project for classifying **Gastrointestinal (GI) conditions** using advanced deep learning and gradient boosting techniques.

---

## 📌 Overview

This repository contains implementations of multiple state-of-the-art approaches for GI classification tasks. The project combines hybrid neural network architectures with ensemble methods to achieve high accuracy, robustness, and generalization in medical image classification.

---

## 🗂️ Project Structure

The repository is organized into two primary approaches:

### 1️⃣ EffNet–MobileViT Hybrid

A hybrid deep learning approach combining:

* **EfficientNet** – Efficient convolutional neural networks balancing accuracy and computational efficiency.
* **MobileViT** – A mobile-friendly vision transformer architecture for improved feature extraction and global attention.

🔹 Designed for strong performance with computational efficiency.

---

### 2️⃣ Swin Transformer + CatBoost Ensemble

An ensemble-based approach combining:

* **Swin Transformer** – Vision transformer with shifted window attention mechanism for capturing complex spatial patterns.
* **CatBoost** – Gradient boosting framework optimized for categorical features and robust against overfitting.

🔹 Designed for maximum predictive performance and robustness.

---

## 🛠️ Technologies Used

* **Programming Language:** Python
* **Implementation Format:** Jupyter Notebooks
* **Deep Learning Frameworks:** PyTorch / TensorFlow (vision models)
* **Ensemble Learning:** CatBoost
* **Development Environment:** Jupyter Notebook

---

## ✨ Features

* Implementation of multiple state-of-the-art model architectures
* Hybrid and ensemble modeling strategies
* Structured and reproducible Jupyter notebooks
* Comparative architectural analysis
* Designed for medical image classification tasks

---

## 🚀 Getting Started

### ✅ Prerequisites

* Python 3.8+
* Jupyter Notebook
* PyTorch or TensorFlow
* Required dependencies (listed inside individual notebooks)

---

### 📥 Installation

1. Clone the repository:

```bash
git clone https://github.com/Longclaw4/GI-Classification.git
cd GI-Classification
```

2. Install required packages (refer to individual notebooks for specific dependencies).

3. Navigate to your preferred approach:

* **Hybrid EfficientNet–MobileViT:** `Effnet Mobvit Hybrid/`
* **Swin–CatBoost Ensemble:** `swin catboost/`

---

## ▶️ Usage

Open the Jupyter notebooks in your selected directory and follow the step-by-step implementation.

Each approach includes notebooks covering:

* **Data Preparation** – Load and preprocess GI classification dataset
* **Model Training** – Train selected architecture
* **Evaluation** – Assess performance using relevant metrics
* **Prediction** – Generate predictions on new samples

---

## 🎯 Project Goals

* Develop accurate GI classification models using modern architectures
* Explore architectural combinations for optimal generalization
* Compare hybrid neural networks with ensemble methods
* Provide reproducible and well-documented implementations
* Enable practical usage for medical imaging classification tasks
* Achieve strong accuracy while maintaining computational efficiency
* Build a foundation for future research and improvements

---

## 🧠 Model Architectures

### 🔹 EfficientNet–MobileViT Hybrid

* Combines EfficientNet efficiency with MobileViT attention mechanisms
* Suitable for resource-constrained environments
* Balanced trade-off between accuracy and computational cost

---

### 🔹 Swin Transformer–CatBoost Ensemble

* Utilizes shifted window attention for strong spatial feature learning
* Applies CatBoost for robust ensemble-based predictions
* Effective for handling complex medical imaging patterns

---

## ⚡ Performance Considerations

* **EfficientNet–MobileViT:** Optimized for faster inference (mobile/edge deployment)
* **Swin–CatBoost:** Optimized for maximum predictive accuracy and ensemble robustness

---

## 📊 Dataset Requirements

The models expect medical imaging data related to gastrointestinal classification.

Ensure your dataset includes:

* Properly labeled images
* Consistent image dimensions
* Balanced class distribution (recommended)
* Train / validation / test split

---

## 📈 Results & Evaluation

Refer to individual notebook files for:

* Detailed evaluation metrics
* Performance comparisons
* Experimental analysis between both approaches

---

## 📌 Notes

This repository is intended for research, experimentation, and educational purposes in medical image classification.
