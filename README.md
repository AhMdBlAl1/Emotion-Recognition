# 🎭 Advanced Facial Emotion Recognition System using Spatial Attention CNN

An enterprise-grade, end-to-end Deep Learning system designed to detect human faces from images and classify their emotional states into seven distinct categories in real-time. Built on top of **PyTorch** for model development and **Gradio** for a sleek, interactive web interface, this repository combines spatial attention mechanisms with convolutional neural networks to achieve state-of-the-art diagnostic visualization.

---

## 📝 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Deep Learning Architecture](#-deep-learning-architecture)
   - [Spatial Attention Mechanism](#spatial-attention-mechanism)
   - [EmotionCNN Architectural Breakdown](#emotioncnn-architectural-breakdown)
4. [Inference & Image Processing Pipeline](#-inference--image-processing-pipeline)
5. [Repository Structure](#-repository-structure)
6. [Installation & Environment Setup](#-installation--environment-setup)
7. [Training and Evaluation Protocol](#-training-and-evaluation-protocol)
8. [Usage & Deployment Guide](#-usage--deployment-guide)
9. [Future Roadmap](#-future-roadmap)

---

## 🎯 Project Overview

Facial Emotion Recognition (FER) plays a pivotal role in human-computer interaction, smart surveillance, mental health monitoring, and personalized user experiences. However, variations in illumination, head poses, and subtle structural facial shifts make accurate classification challenging. 

This project addresses these challenges by embedding a custom **Spatial Attention Module** within a deep Convolutional Neural Network (`EmotionCNN`). The model isolates high-priority facial landmarks (such as the brow contours, mouth curvature, and ocular regions) while filtering out redundant background noise. The final model classifies 48x48 pixel grayscale face fields into **7 cardinal emotions**:
* 🤬 **Angry**
* 🤢 **Disgust**
* 😨 **Fear**
* **Happy**
* 😐 **Neutral**
* 😢 **Sad**
* 😲 **Surprise**

---

## ✨ Key Features

* **Attention-Driven Core:** Integrates a learnable 2D Spatial Attention block that dynamically weights feature maps based on feature relevance.
* **Automated Face Cascades:** Features built-in face localization using OpenCV's Haar Cascade Frontal Face algorithm, extracting regions of interest (ROI) instantly before passing them to the neural pipeline.
* **Production-Ready Web App:** Implements a declarative UI using `Gradio Blocks`, rendering side-by-side comparative views of the cropped face and full probability histograms for the 7 emotion classes.
* **Extensive Metrics Logging:** The training workflow natively supports metrics tracking including Multi-class Confusion Matrices and detailed Scikit-Learn Classification Reports (Precision, Recall, F1-Score).
* **Hardware Agnostic:** Automatically binds to NVIDIA CUDA kernels if available, smoothly falling back to CPU execution lines for deployment compatibility.

---

## 🧠 Deep Learning Architecture

### Spatial Attention Mechanism
Standard CNNs treat all spatial regions with equal importance. The `Attention` block used here acts as a feature gating mechanism:

$$\text{Output} = X \times \sigma(W \times X)$$

Where $\sigma$ represents the Sigmoid activation, and $W$ represents a $1 \times 1$ 2D Convolution filter that squashes channel depth into a unified spatial mask. This enables the model to amplify highly expressive facial regions and suppress static non-facial zones.

```text
Input Feature Map (C x H x W) ───┬─────────────────────────────► [ Element-Wise ] ──► Scaled Output
                                 │                                 Multiply  
                                 └──► [Conv2d 1x1] ──► [Sigmoid] ──────┘
                                     (Attention Weights Mask)
