# STINK 3014 Neural Networks - Assignment #2

## 📄 Overview
This repository contains the practical implementation and analysis for **Assignment #2** of the **STINK 3014 (Neural Networks)** course. The project involves building a **Multilayer Perceptron (MLP)** from scratch using Python to understand the mechanics of **Backpropagation** and **Momentum**.

## 🧠 Model Architecture
* **Type**: Feedforward Neural Network (Single Hidden Layer)
* **Inputs**: 2 ($X1, X2$)
* **Hidden Units**: 1
* **Output Units**: 1
* **Activation Function**: Sigmoid (used in both hidden and output layers).

## 📂 Files Included
* **`Assignment 2.py`**: Python source code implementing Backpropagation with Momentum.
* **`STINK3014-A251-Assignment-#2.docx`**: Report containing theoretical answers and experimental findings.
* **`ASSIGNMENT 2 INSTRUCTION.pdf`**: Original assignment requirements.

## 🧪 Key Experiments
* **Learning Rate ($\alpha$)**: Analyzed convergence speed with rates 0.1, 0.5, and 0.9.
* **Momentum ($\beta$)**: Observed how momentum (0.99) speeds up learning and escapes local minima.
* **Initial Weights**: Tested how negative weights affect early training epochs.

## 🚀 Usage
```bash
python "Assignment 2.py"