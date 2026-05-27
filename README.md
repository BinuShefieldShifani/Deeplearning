# 🧠 Deep Learning — From Foundations to Advanced Architectures

> A progressive notebook series covering deep learning from tensor mathematics through CNNs, transfer learning, regularisation, adversarial attacks, and TensorBoard diagnostics — implemented in TensorFlow/Keras with full training logs.

---

## 📌 Overview

This repository documents hands-on deep learning experiments built as part of the **MS AI & Automation** programme at University West, Sweden. Each notebook is a self-contained, well-commented exploration — designed to build genuine understanding before reaching for library abstractions.

The series spans five topics: mathematical foundations → training mechanics → convolutional networks → regularisation & robustness → diagnostics & optimisation.

---

## 📂 Notebook Series

| # | File | Topic | Key Results |
|---|------|--------|-------------|
| 01 | [`01_tensors_pca_neural_networks_fundamentals.ipynb`](#) | Tensors, PCA, Feedforward NNs | Manual SVD matches sklearn PCA exactly; KernelPCA pipeline with GridSearchCV |
| 02 | [`02_classification_regression_optimizers.ipynb`](#) | CNN classifiers, RNNs, Optimiser comparison | MNIST CNN: **98.4%** · IMDb sentiment: **87.3%** · Fashion-MNIST transfer (MobileNetV2): **91.0%** |
| 03 | [`03_cnn_architectures_transfer_learning.ipynb`](#) | Custom convolution, FCN vs CNN, ResNet50, Xception | Xception on tf_flowers: **89.1% val accuracy** in 5 epochs |
| 04 | [`04_regularisation_multitask_adversarial.ipynb`](#) | L1/L2, Dropout, multi-task learning, adversarial examples | CIFAR-10 multi-task: **76.3% multi-class + 92.2% binary** simultaneously |
| 05 | [`05_learning_rate_vanishing_gradients_tensorboard.ipynb`](#) | LR finder, vanishing gradients, TensorBoard | 100-layer activation comparison: sigmoid collapses, SELU/ELU remain stable |

---

## 🔑 Concepts Covered

**Mathematics & Representations**
- Tensor operations: slicing, broadcasting, stacking, splitting
- PCA from scratch via SVD — verified against `sklearn.decomposition.PCA`
- KernelPCA (linear, RBF, sigmoid kernels) with hyperparameter search

**Training Mechanics**
- Optimiser comparison: SGD · SGD+Momentum · Nesterov · AdaGrad · RMSProp · Adam · Nadam
- Learning rate finder (log-sweep 1e-6 → 1e-3, loss vs LR plot)
- EarlyStopping, ModelCheckpoint, LR scheduling

**Convolutional Networks**
- Convolution and max/average pooling implemented from scratch with NumPy
- FCN vs CNN comparison on Fashion-MNIST (CNN wins: **90.5% vs 88.9%**)
- Multi-task CNN: simultaneous 8-class + binary classification on CIFAR-10

**Transfer Learning**
- MobileNetV2 (frozen backbone) → tf_flowers: **91% val accuracy**
- Xception (frozen backbone) → tf_flowers: **89.1% val accuracy** in 5 epochs
- ResNet50 architecture instantiation and fine-tuning setup

**Regularisation & Robustness**
- Dropout + L1/L2 + MaxNorm on MNIST: **98.3% test accuracy**
- Ensemble decision trees: accuracy vs ensemble size analysis
- Adversarial examples: non-targeted and targeted attacks on MNIST classifier

**Diagnostics**
- Vanishing gradient study: 100-layer networks with sigmoid vs ReLU vs ELU vs SELU
- TensorBoard: loss curves, weight histograms, gradient tracking
- Custom `CustomAugmentationLayer` (brightness, saturation, hue, contrast)

---

## 📊 Key Results Summary

| Task | Dataset | Model | Test Accuracy |
|------|---------|-------|--------------|
| Digit classification | MNIST | CNN + Dropout | **98.4%** |
| Fashion classification | Fashion-MNIST | CNN | **90.5%** |
| Sentiment analysis | IMDb | Embedding + Dense | **87.3%** |
| Flower classification | tf_flowers | MobileNetV2 (transfer) | **91.0%** |
| Flower classification | tf_flowers | Xception (transfer) | **89.1%** |
| Housing price regression | California Housing | Dense NN | MAE **0.349** |
| Multi-task (8-class + binary) | CIFAR-10 | Custom CNN | **76.3% / 92.2%** |
| Regularised classification | MNIST | Dense + L1/L2 + Dropout | **98.3%** |


## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/BinuShefieldShifani/Deeplearning.git
cd Deeplearning
```

### 2. Install dependencies

```bash
pip install tensorflow==2.11 scikit-learn numpy matplotlib opencv-python tensorflow-datasets
```

### 3. Launch Jupyter

```bash
jupyter notebook
```

Open notebooks in order (01 → 05) for the full learning progression, or jump directly to any topic.



---

## 🗂️ Repository Structure

```
Deeplearning/
├── 01_tensors_pca_neural_networks_fundamentals.ipynb   # Tensors, PCA, feedforward NN
├── 02_classification_regression_optimizers.ipynb       # Classifiers, regression, optimiser study
├── 03_cnn_architectures_transfer_learning.ipynb        # CNNs from scratch, ResNet, Xception
├── 04_regularisation_multitask_adversarial.ipynb       # Regularisation, multi-task, adversarial
├── 05_learning_rate_vanishing_gradients_tensorboard.ipynb  # LR finder, TensorBoard, gradients
└── README.md
```



---

## 🔬 Design Highlights

- **Scratch-first approach** — convolution, pooling, and PCA are implemented manually before using library versions, building real intuition about what the abstractions do.
- **Controlled comparisons** — optimisers, activations, and architectures are benchmarked side-by-side on identical tasks so differences are attributable.
- **Adversarial robustness** — non-targeted and targeted perturbations are generated and visualised, showing how small input changes fool a 98%+ accurate classifier.
- **Diagnostic tooling** — TensorBoard integration with histogram tracking lets you observe weight distributions and gradient flow epoch-by-epoch.

---

## 🔗 Related Repositories

- [**MarineDebris**](https://github.com/BinuShefieldShifani/MarineDebris) — DeepLabV3+ semantic segmentation of underwater plastic pollution (Dice: 0.717, IoU: 0.611)


---

## 👤 Author

**Binu Shefield Shifani**

Software Engineer · 5 years at Cognizant Technology Solutions  
MS AI & Automation · University West, Trollhättan, Sweden  
[GitHub](https://github.com/BinuShefieldShifani)
