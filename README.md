# Neural Network Pruning Strategies for Time-Series Forecasting

This repository contains a specialized research framework for **Hidden Layer Pruning** methodologies applied to Energy Consumption Forecasting. The core objective is to reduce the computational footprint of Deep Learning models while preserving their predictive power through innovative weight reconfiguration algorithms.

---

## 💻 Technical Environment

The project is optimized for the following stack:
* **Python:** 3.11.5
* **TensorFlow:** 2.17.0
* **Keras:** 3.11.3 (Leveraging the Keras 3 unified backend)
* **Hardware Recommendation:** Execution on GPU is supported but not mandatory due to the optimized pruning loops.

---

## 📂 Repository Structure

The project is structured into two specialized Jupyter Notebooks:

### 1. `Pruning_Strategies.ipynb`
**The Modular Algorithm Library.**
* **Pure Logic:** Contains the standalone Python functions for each pruning strategy.
* **Ideal for Research:** Use this notebook to inspect how weights are mathematically mapped and transformed during the pruning process without the overhead of the full training cycle.

---

## 🛠️ Pruning Methodologies Explained

Traditional pruning often results in "broken" networks. This project solves this by using weight redistribution:

### A. Backward Strategy (BS)

This strategy reconfigures the network by looking "backward" from the layer targeted for removal. It uses **absolute mean similarity** to map the importance of neurons in the target layer back to the preceding layer. This ensures that the features learned by the pruned layer are "absorbed" by its predecessor.

### B. Forward Strategy (FS)
The Forward Strategy focuses on the interface between the pruned layer and the subsequent one. It adjusts the input weight matrix of the "next" layer to accommodate the new input dimensions coming from the layer before the pruned one.

---

## 🚀 Getting Started

### Installation
Ensure your environment matches the required versions to avoid API conflicts:
```bash
pip install tensorflow==2.17.0 keras==3.11.3 pandas numpy scikit-learn matplotlib
