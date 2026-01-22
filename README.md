# ATLAS OpenData DNN Classification

This project implements a **Deep Neural Network (DNN)** pipeline for **binary classification** of high-energy physics events using **ATLAS OpenData**.  
The goal is to distinguish **signal** events (Higgs boson production with leptonic decays) from **background** events (top–antitop production).

The work follows the **ATLAS DNN tutorial**, with a clear separation between:
- **Exercise 5-1:** Structuring the machine learning pipeline
- **Exercise 5-2:** Improving the model with modern training techniques

---

## Project Overview

**Physics processes**
- **Signal:** Higgs → WW → leptons  
- **Background:** tt̄ → leptons  

**Machine learning task**
- Binary classification (signal vs background)
- Fully connected Deep Neural Network
- Implemented with **PyTorch**

**Key idea**
- ROOT is used **only during data preprocessing**
- After preprocessing, the workflow is **ROOT-free**
- Processed data are stored in **HDF5** format for fast reuse

---

## Repository Structure

```text
.
├── prepare_data.py        # ROOT → selections → HDF5 (Exercise 5-1)
├── train.py               # DNN training (Exercise 5-1 & 5-2)
├── evaluate.py            # Model evaluation (accuracy, ROC AUC)
├── study_results.py       # Score distributions and ROC plots
├── data/
│   └── atlas_processed.h5 # Processed dataset (generated)
├── models/
│   └── dnn_best.pt        # Best trained model (generated)
├── plots/
│   ├── scores.png
│   ├── roc.png
│   ├── scores5-2.png
│   ├── roc5-2.png
│   └── loss.png
└── README.md
