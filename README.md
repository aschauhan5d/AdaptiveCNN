
# Adaptive CNN – Automatically Growing Neural Network

## 🔍 Overview
This project implements an **Adaptive Convolutional Neural Network (Adaptive CNN)** that **automatically increases its depth**
based on validation performance during training.

Unlike traditional CNNs with fixed architectures, this model **dynamically grows** when learning stagnates, making it a
lightweight and research-inspired alternative to Neural Architecture Search (NAS).

---

## 🚀 Key Features
- Dynamic CNN architecture (auto layer growth)
- Validation-loss–driven decision making
- Simple AutoML-inspired design
- Clean, research-ready implementation

---

## 🧠 Core Idea
1. Start with a shallow CNN  
2. Train for a few epochs  
3. Monitor validation loss  
4. If improvement < threshold → add Conv layer  
5. Repeat until max depth or convergence  

---

## 📁 Repository Structure
```
adaptive-cnn/
│── model/
│   └── adaptive_model.py
│── train.py
│── requirements.txt
│── results/
│── README.md
```

---

## 📦 Dataset
- **CIFAR-10** (automatically downloaded via TensorFlow)

---

## 🛠️ Installation
```bash
pip install -r requirements.txt
```

---

## ▶️ Run Training
```bash
python train.py
```

---

## 📊 Sample Output
```
Step 1 | Val Loss: 1.25
Step 2 | Val Loss: 1.04
No improvement → Adding new Conv layer
Step 3 | Val Loss: 0.97
```

---

## 📌 Research Relevance
- AutoML concepts
- Adaptive systems
- Model capacity control
- Neural architecture exploration

---

## 👤 Author
Abhishek Singh Chauhan  
MCA (AI/ML)

---

## 📜 License
MIT License
