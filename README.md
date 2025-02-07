
# Federated Learning with Flower  


A privacy-preserving machine learning implementation using federated learning with the Flower framework. This project demonstrates collaborative model training across multiple clients without sharing raw data.  

### **Key Technologies**  
-  **Flower** - Federated Learning Framework  
-  **PyTorch** - Deep Learning Library  
-  **Hydra** - Configuration Management  
-  **CML** - Continuous Machine Learning  

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
├── github/
│   └── workflows/
│       └── cml.yaml      # CI/CD workflow definition
├── pyecache/              # Python cache directory
├── data/                 # Dataset files
├── plot/                 # Visualization outputs
├── client.py             # Flower client logic
├── client_utils.py       # Client helper functions
├── dataset.py            # Data loading/preprocessing
├── poetry.lock           # Poetry dependency lockfile
├── pyproject.toml        # Poetry project configuration
├── requirements.txt      # Python dependencies
├── run.py                # Main execution script
├── run_bagging.sh        # Bagging experiment script
├── run_cyclic.sh         # Cyclic experiment script
├── server.py             # Flower server logic
├── server_utils.py       # Server helper functions
├── sim.py                # Start simulation
├── utils.py              # Shared utilities
└── README.md             # Project documentation
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
---

# Comparison of Federated XGBoost Strategies: Cyclic vs. Bagging

A comparison of two federated learning strategies for XGBoost implementations using the Flower framework.

## 🔄 **FedXgbCyclic**
**Documentation**: [flwr.server.strategy.FedXgbCyclic](https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedXgbCyclic.html)

### Key Characteristics:
- **Client Selection**: Sequential cycling through clients in fixed order
- **Training Pattern**: One client per round, sequential execution
- **Data Requirements**: Effective for non-IID data distributions
- **Tree Growth**: Builds trees sequentially across clients
- **Aggregation**: Maintains global model that cycles through clients
- **Use Case**: Client-ordered scenarios where data sequence matters

## 🎒 **FedXgbBagging**
**Documentation**: [flwr.server.strategy.FedXgbBagging](https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedXgbBagging.html)

### Key Characteristics:
- **Client Selection**: Random subset selection each round
- **Training Pattern**: Parallel client training (multiple clients per round)
- **Data Requirements**: Works best with IID data distributions
- **Tree Growth**: Builds multiple candidate trees in parallel
- **Aggregation**: Uses bootstrap aggregating (bagging) for ensemble effects
- **Use Case**: Traditional federated scenarios with independent data

## 📊 Key Differences

| Feature                | Cyclic                                  | Bagging                                |
|------------------------|-----------------------------------------|----------------------------------------|
| **Client Selection**   | Fixed order, sequential                 | Random subset, parallel                |
| **Round Execution**    | 1 client/round                          | Multiple clients/round                 |
| **Data Assumption**    | Tolerates non-IID                       | Prefers IID                            |
| **Tree Building**      | Sequential tree growth                  | Parallel tree candidates               |
| **Aggregation**        | Direct model cycling                    | Bootstrap aggregating                  |
| **Communication**      | Low bandwidth (1 client/round)          | Higher bandwidth                       |
| **Use Case**           | Ordered client sequences                | Traditional FL scenarios               |
| **Performance**        | Better for client-specific patterns     | Better for generalizable models        |


## ⚖️ When to Use Which

### Choose **Cyclic** When:
- Clients have ordered/sequential data relationships
- Data distribution is non-IID across clients
- You want explicit client participation order
- Bandwidth is constrained

### Choose **Bagging** When:
- Data is IID or approximately independent
- You want traditional federated averaging behavior
- Parallel client participation is preferred
- Ensemble effects are desirable

---

## Implementation Tips
1. **Cyclic** requires careful client ordering configuration
2. **Bagging** benefits from larger client subsets per round
3. Both support XGBoost's histogram-based training
4. Monitor client compute resources differently:
   - Cyclic: Manage sequential load
   - Bagging: Handle parallel compute demands
---

## Credits
This project uses code adapted from the [Flower XGBoost Comprehensive Example](https://github.com/adap/flower/tree/main/examples/xgboost-comprehensive) as the initial code skeleton.











