
# 🌸 Federated Learning with Flower  


A privacy-preserving machine learning implementation using federated learning with the Flower framework. This project demonstrates collaborative model training across multiple clients without sharing raw data.  

### **Key Technologies**  
- 🌼 **Flower** - Federated Learning Framework  
- 🔥 **PyTorch** - Deep Learning Library  
- 🔧 **Hydra** - Configuration Management  
- 📊 **CML** - Continuous Machine Learning  

## **📚 Table of Contents**
- [✨ Features](#-features)  
- [📂 Project Structure](#-project-structure)  
- [🚀 Getting Started](#-getting-started)  
- [⚙️ Configuration](#configuration)  
- [🧪 Running Experiments](#-running-experiments)  
- [📂 Output Structure](#-output-structure)

---

## ✨ Features  
✅ **Privacy-Preserving Training** - Federated learning implementation with data isolation  
✅ **Flexible Configuration** - Hydra-powered experiment management  
✅ **Reproducible Experiments** - Automatic output organization  
✅ **CI/CD Integration** - GitHub Actions workflow with CML reporting  
✅ **Custom Dataset Support** - CSV data loader with preprocessing pipeline  

---

## **📂 Project Structure**
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
```

---

## **🚀 Getting Started**

### **Prerequisites**  
Before running the project, ensure you have the following installed:  
- Python 3.8+  
- pip (Python package manager)  

### **Installation**  

1. **Clone the repository**  
   ```bash
   git clone https://github.com/moh-a-abde/FL-CML-Pipeline.git
   cd FL-CML-Pipeline
   ```
2. **Create and activate a virtual environment**
   ```bash
   python -m venv fl-env
   source fl-env/bin/activate  # Linux/MacOS
   # On Windows:
   # fl-env\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
---

## **⚙️ Configuration**

The experiment settings are managed using **Hydra** and are defined in `conf/base.yaml`.  
Modify these settings in conf/base.yaml or override them at runtime when executing experiments.

Here are the key parameters:  

```yaml
# Core Experiment Parameters
num_rounds: 10                   # Total training rounds
num_clients: 100                 # Total available clients
batch_size: 20                   # Local batch size
num_classes: 2                   # Output classes

# Client Sampling
num_clients_per_round_fit: 10    # Clients per training round
num_clients_per_round_eval: 25   # Clients per evaluation round

# Training Configuration
config_fit:
  lr: 0.01                       # Learning rate
  momentum: 0.9                  # SGD momentum
  local_epochs: 1                # Epochs per client update
```

---

### **🧪 Running Experiments**  

### Basic Execution  
To start a federated learning simulation with default settings:  
```python
python main.py
```
### Run with Custom Configuration

Override default parameters at runtime:
```python
python main.py num_rounds=5 num_clients=500 config_fit.lr=0.1 config_fit.local_epochs=2
```

---

## **📂 Output Structure**

Experiment outputs are automatically saved in the `outputs/` directory, organized by date and time. Each experiment run generates a unique folder with the following structure:  

```plaintext
outputs/
└── YYYY-MM-DD/                  # Run date
    └── HH-MM-SS/                # Run time
        ├── .hydra/              # Config snapshots
        │   ├── config.yaml
        │   └── hydra.yaml
        └── results.pkl          # Training history
```










