# 🩺 Enhanced SwinCheX: Hybrid Swin Transformer + CNN for ChestX-ray14 with Grad-CAM Explainability

This repository contains **Enhanced SwinCheX**, an improved version of the SwinCheX model for chest disease detection on the [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) dataset.

Our enhancements include:
- **Hybrid CNN + Swin Transformer backbone** for better spatial + contextual feature extraction.
- **Grad-CAM-based explainability** for visualizing model attention.
- **One-hot encoding for multi-label classification**.
- **Test pipeline** that generates prediction metrics, Grad-CAM visualizations, and per-class summaries.

---

## 📌 Features
✅ **Hybrid Architecture** — Combines a CNN feature extractor with Swin Transformer blocks.  
✅ **Multi-Label Learning** — Handles 14 chest diseases with one-hot encoded labels.  
✅ **Explainability** — Grad-CAM heatmaps for model interpretability.  
✅ **Folder-Based Inference** — Runs predictions on all images in a given folder.  
✅ **Customizable Backbone** — Easily swap CNN base (e.g., ResNet, DenseNet).

---

## Model Architecture

[Input Image]
     ↓
[ResNet50 CNN Backbone]
     ↓
[Swin Transformer Blocks]
     ↓
[Global Pooling]
     ↓
[Classification Head (14 Sigmoid outputs)]



