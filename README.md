# Federated Learning with Flower  

A privacy-preserving machine learning implementation using federated learning with the Flower framework. This project demonstrates collaborative model training across multiple clients without sharing raw data.  

### **Key Technologies**  
-  **Flower** - Federated Learning Framework  
-  **PyTorch** - Deep Learning Library  
-  **Hydra** - Configuration Management  
-  **CML** - Continuous Machine Learning

---

## 🛠️ Workflow Overview

```diff
+============================================[ DATA PIPELINE ]============================================+
!                                                                                                         !
!  1. Live Network Capture → 2. Clean Capture and Convert to Dataset → 3. Train/Test → 4.️Output Results   !
!                                                                                                         !
+=========================================================================================================+
```

---

## 🗺️ Architecture Overview

This library implements a federated learning system that:
1. Processes network traffic data
2. Trains an XGBoost model in a distributed manner
3. Detects network intrusions across multiple clients while preserving data privacy

The system consists of several key components:

1. Data Processing Pipeline

- `data/livepreprocessing_socket.py`: Processes live network traffic data from Kafka
- `data/receiving_data.py`: Receives and saves processed data
- `dataset.py`: Handles data loading, preprocessing, and partitioning

2. Federated Learning Core

- `server.py`: Central FL server implementation
- `client.py`: FL client implementation
- `client_utils.py`: Client-side helper functions and XGBoost client class
- `server_utils.py`: Server-side helper functions and client management

3. Training Methods

Two main training approaches:
- Bagging: Aggregates models from multiple clients
- Cyclic: Passes model sequentially through clients

4. Execution Scripts

- `run_bagging.sh`: Launches bagging-based training
- `run_cyclic.sh`: Launches cyclic training
- `run.py`: Orchestrates the entire training pipeline
- `sim.py`: Simulation environment for testing

---

## 🎯 What is to be achieved?

1. Data Processing
- Real-time data ingestion from Kafka
- Automated preprocessing of network traffic data
- Support for multiple feature types (categorical and numerical)
- Dynamic data partitioning across clients

2. Model Training
- Distributed XGBoost training
- Support for both bagging and cyclic training methods
- Configurable local training rounds
- Centralized and decentralized evaluation options

3. Scalability & Configuration
- Configurable number of clients and rounds
- Adjustable learning rates and model parameters
- Support for CPU/GPU training
- Flexible client selection strategies

4. Evaluation & Metrics
- Support for multiple evaluation metrics:
  - Precision
  - Recall
  - F1 Score
- Centralized and distributed evaluation options
  
---

## **📚 Table of Contents**
- [✨ Features](#-features)  
- [📂 Project Structure](#-project-structure)  
- [🚀 Getting Started](#-getting-started)  
- [⚙️ Configuration](#-configuration)
- [📂 Output Structure](#-output-structure)
- [🧪 Running Experiments](#-running-experiments)  
- [⚖️ Comparison of Federated XGBoost Strategies: Cyclic vs. Bagging](#-comparison-of-federated-xgboost-strategies:-cyclic-vs.-bagging)

---

## ✨ Features  
✅ **Privacy-Preserving Training** - Federated learning implementation with data isolation  
✅ **Flexible Configuration** - Hydra-powered experiment management  
✅ **Reproducible Experiments** - ⚠️Automatic output organization   
✅ **CI/CD Integration** - GitHub Actions workflow with CML reporting  
✅ **Custom Dataset Support** - CSV data loader with preprocessing pipeline  

---

## **📂 Project Structure**
```bash
├── github/
│   └── workflows/
│       └── cml.yaml      # CI/CD workflow definition
├── pyecache/             # Python cache directory
├── data/                 # Dataset files, data capture script, and data cleaning script
├── plot/                 # Visualization outputs - 🚧 under construction (implementation phase) 🚧
├── client.py             # Flower client logic
├── client_utils.py       # Client helper functions
├── dataset.py            # Data loading/preprocessing
├── poetry.lock           # Poetry dependency lockfile - 🔍 exploring (research phase) 🔍
├── pyproject.toml        # Poetry project configuration - 🔍 exploring (research phase) 🔍
├── requirements.txt      # Python dependencies
├── run.py                # runs FULL FULL & CML experiment; includes capturing data traffic and preprocessing - 🚧 under construction (implementation phase) 🚧
├── run_bagging.sh        # Bagging experiment script - runs script.py + client.py
├── run_cyclic.sh         # Cyclic experiment script - runs script.py + client.py
├── server.py             # Flower server logic
├── server_utils.py       # Server helper functions
├── sim.py                # Start simulation - ⚠️ deprecated soon ⚠️
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
2. **Create and activate a virtual environment (Docker is being used to run CML locally to automate the workflow)**
   **After setting up the docker environment run the following:**
   ```bash
   sudo systemctl start docker
   sudo systemctl enable docker
   act -j run --container-architecture linux/amd64 -v
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
---

## **⚠️⚙️ Configuration**

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

## **⚠📂 Output Structure**

Experiment outputs are automatically saved in the `outputs/` directory, organized by date and time. Each experiment run generates a unique folder with the following structure:  

```plaintext
outputs/
└── YYYY-MM-DD/                  # Run date
    └── HH-MM-SS/                # Run time
        ├── .hydra/              # ⚠️Config snapshots
        │   ├── config.yaml
        │   └── hydra.yaml
        ├── results.pkl          # Training history
        ├── predictions/         # Model predictions
            ├── predictions_round_X.csv  # Per-round predictions


```

All these files are automatically tracked by the CML workflow and included in result reports.

---

### **🧪 Running Experiments**  

### Basic Execution  
To start federated learning with default settings:  
```bash
./run_bagging.sh
```
or

```bash
./run_bagging.sh
```

---

# ⚖️ Comparison of Federated XGBoost Strategies: Cyclic vs. Bagging

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

## When to Use Which

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

---

<!-- ༼ つ ◕_◕ ༽つ R&D ZONE ༼ つ ◕_◕ ༽つ -->
<div align="center">

## 🔥 **R&D Led By** 🔥
### [ **`Mohamed Abdel-Hamid`** ]

![Static Badge](https://img.shields.io/badge/Phase-%F0%9F%94%A5_Innovation_Station-%23FF6B6B?style=for-the-badge)
<br>

```diff
+==================================================+
!  🧑💻 Coded with 100% chaos-driven curiosity    !
!  ☕ Powered by midnight espresso & big dreams   !
+==================================================+
```
<sub>
🔐 Cyber Alchemy Brewing For 🏛️ Indiana University of Pennsylvania's ARMZTA Project

🔗 https://www.iup.edu/cybersecurity/grants/ncae-c-armzta/index.html</sub>

<sub>Grant: NCAE-C Program</sub>

</div> 
<!-- ༼ つ ◕_◕ ༽つ R&D ZONE ༼ つ ◕_◕ ༽つ --> 









