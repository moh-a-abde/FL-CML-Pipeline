# 🌸 Federated Learning with Flower  


A privacy-preserving machine learning implementation using federated learning with the Flower framework. This project demonstrates collaborative model training across multiple clients without sharing raw data.  

### **Key Technologies**  
- 🌼 **Flower** - Federated Learning Framework  
- 🔥 **PyTorch** - Deep Learning Library  
- 🔧 **Hydra** - Configuration Management  
- 📊 **CML** - Continuous Machine Learning  

## 📚 Table of Contents  
- [✨ Features](#-features)  
- [📂 Project Structure](#-project-structure)  
- [🚀 Getting Started](#-getting-started)  
- [⚙️ Configuration](#-configuration)  
- [🧪 Running Experiments](#-running-experiments)  
- [🔧 Customization Guide](#-customization-guide)  


---

## ✨ Features  
✅ **Privacy-Preserving Training** - Federated learning implementation with data isolation  
✅ **Flexible Configuration** - Hydra-powered experiment management  
✅ **Reproducible Experiments** - Automatic output organization  
✅ **CI/CD Integration** - GitHub Actions workflow with CML reporting  
✅ **Custom Dataset Support** - CSV data loader with preprocessing pipeline  

---

## 📂 Project Structure  
```bash
├── conf/                 # Hydra configurations
│   └── base.yaml         # Main experiment settings
├── GitHub/
│   └── workflows/        # CI/CD pipelines
│       └── cml.yaml      # ML workflow definition
├── outputs/              # Experiment outputs
├── client.py             # Flower client logic
├── dataset.py            # Data loading/preprocessing
├── main.py               # Entry point with Hydra
├── model.py              # Neural network architecture
├── server.py             # Flower server utilities
├── requirements.txt      # Dependencies
└── README.md             # You are here 📍
