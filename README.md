# DFFNN Pruning Strategies

This repository contains a specialized research framework for **Hidden Layer Pruning** methodologies in Deep Feedforward Neural Networks (DFFNNs). The core objective is to reduce the computational footprint of deep learning models while preserving their predictive power through innovative weight redistribution algorithms.

---

## 💻 Technical Environment

The project is optimized for the following stack:
- **Python:** 3.11.5
- **TensorFlow:** 2.17.0
- **Keras:** 3.11.3 (Leveraging the Keras 3 unified backend)

---

## 📂 Repository Structure

The project features a clean, modular structure containing the core implementation:

- **`Pruning_Strategies.ipynb`** (or main script): Contains all necessary library imports and the standalone functions implementing the **Backward Strategy (BS)** and **Forward Strategy (FS)** layer-level pruning algorithms.

---

## 🛠️ Pruning Methodologies Explained

Traditional pruning often disrupts network continuity. This framework solves this by utilizing smart weight redistribution mechanisms:

### A. Backward Strategy (BS)
This strategy reconfigures the network by looking "backward" from the layer targeted for removal. It maps the importance of neurons in the target layer back to the preceding layer, ensuring that the features learned by the pruned layer are efficiently absorbed by its predecessor.

### B. Forward Strategy (FS)
The Forward Strategy focuses on the interface between the pruned layer and the subsequent one. It adjusts the input weight matrix of the next layer to seamlessly accommodate the new dimensional flow resulting from the removal.

---

## 🚀 Getting Started

### Installation
Ensure your environment matches the required versions to avoid API conflicts:
```bash
pip install tensorflow==2.17.0 keras==3.11.3 pandas numpy scikit-learn matplotlib
