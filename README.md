# HFT Quantum Machine Learning Project

A comprehensive project exploring the application of Quantum Machine Learning (QML) techniques to High-Frequency Trading (HFT) strategies. This project compares classical machine learning approaches with quantum-enhanced models for financial market prediction and trading signal generation.

## 🎯 Project Overview

This project investigates whether quantum machine learning can provide advantages over classical methods in high-frequency trading applications. We implement and compare various quantum algorithms including Variational Quantum Classifiers (VQC), Quantum Neural Networks (QNNs), and Quantum Support Vector Machines against traditional ML models.

### Key Features

- **Quantum Data Encoding**: Amplitude and angle encoding techniques for financial data
- **Hybrid Models**: Classical-quantum hybrid architectures
- **Performance Comparison**: Comprehensive benchmarking of quantum vs classical approaches
- **Real-time Prediction**: Live trading signal generation system
- **Interactive Demo**: Web-based interface for model testing

## 📁 Project Structure

```
HFT-QML-Project/
│
├── 📄 README.md                     # This file
├── 📄 requirements.txt              # Python dependencies
├── 📄 .gitignore                   # Git ignore rules
│
├── 📂 data/                        # Dataset storage
│   ├── raw/                        # Original market data (CSV, JSON)
│   ├── processed/                  # Cleaned & quantum-encoded data
│   └── data_description.md         # Dataset documentation
│
├── 📂 notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb   # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb # Feature creation & quantum encoding
│   ├── 03_model_training.ipynb     # Model training & hyperparameter tuning
│   ├── 04_results_visualization.ipynb # Performance analysis & plots
│   └── 05_quantum_circuit_tests.ipynb # Quantum circuit validation
│
├── 📂 src/                         # Source code modules
│   ├── __init__.py
│   ├── config.py                   # Configuration & hyperparameters
│   ├── data_preprocessing.py       # Data cleaning & normalization
│   ├── quantum_encoding.py         # Quantum state encoding methods
│   ├── quantum_models.py           # QML model implementations
│   ├── classical_models.py         # Classical ML baselines
│   ├── evaluation.py               # Model evaluation metrics
│   ├── utils.py                    # Utility functions
│   └── predict.py                  # Prediction pipeline
│
├── 📂 models/                      # Trained model storage
│   ├── classical/                  # Classical ML models
│   └── quantum/                    # Quantum ML models
│
├── 📂 results/                     # Experimental results
│   ├── metrics.json                # Performance metrics
│   ├── plots/                      # Visualization outputs
│   └── comparisons.md              # Analysis summaries
│
├── 📂 live_demo/                   # Interactive demo
│   ├── app.py                      # Streamlit/Flask application
│   ├── templates/                  # HTML templates
│   └── static/                     # CSS, JS, images
│
└── 📂 docs/                        # Documentation
    ├── literature_review.md
    ├── methodology.md
    ├── results.md
    ├── discussion.md
    └── references.bib
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- IBM Quantum Experience account (optional, for real quantum hardware)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/HFT-QML-Project.git
   cd HFT-QML-Project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up quantum environment (optional)**
   ```bash
   # For IBM Quantum
   pip install qiskit-ibm-runtime
   # Configure your IBM Quantum token
   python -c "from qiskit_ibm_runtime import QiskitRuntimeService; QiskitRuntimeService.save_account('YOUR_TOKEN')"
   ```

### Quick Demo

1. **Start with data exploration**
   ```bash
   jupyter notebook notebooks/01_data_exploration.ipynb
   ```

2. **Run the complete pipeline**
   ```bash
   python src/predict.py --mode train --model all
   ```

3. **Launch interactive demo**
   ```bash
   cd live_demo
   streamlit run app.py
   ```

## 💻 Usage

### Training Models

Train individual models:
```bash
# Classical models
python src/predict.py --mode train --model classical

# Quantum models
python src/predict.py --mode train --model quantum

# All models
python src/predict.py --mode train --model all
```

### Making Predictions

Generate predictions on new data:
```bash
# Using best performing model
python src/predict.py --mode predict --data data/processed/test_data.csv

# Using specific model
python src/predict.py --mode predict --model vqc --data data/processed/test_data.csv
```

### Evaluation

Compare model performance:
```bash
python src/evaluation.py --compare all --output results/comparison_report.html
```

## 🔬 Quantum Models Implemented

### 1. Variational Quantum Classifier (VQC)
- Parameterized quantum circuits for classification
- Gradient-based optimization
- Feature maps: RY, RZ rotation gates

### 2. Quantum Neural Networks (QNN)
- Quantum analogs of classical neural networks
- Trainable quantum layers
- Hybrid classical-quantum architectures

### 3. Quantum Support Vector Machine (QSVM)
- Quantum kernel methods
- Feature mapping to higher-dimensional Hilbert spaces
- Quantum advantage in kernel computation

## 📊 Data Encoding Techniques

### Amplitude Encoding
```python
# Encode classical data into quantum amplitudes
from src.quantum_encoding import amplitude_encoding
quantum_state = amplitude_encoding(classical_features)
```

### Angle Encoding
```python
# Encode features as rotation angles
from src.quantum_encoding import angle_encoding
quantum_circuit = angle_encoding(features, num_qubits=4)
```

## 🔧 Configuration

Modify `src/config.py` for:
- Model hyperparameters
- Quantum circuit parameters
- Data preprocessing settings
- Evaluation metrics

```python
# Example configuration
QUANTUM_CONFIG = {
    'num_qubits': 4,
    'depth': 3,
    'optimizer': 'COBYLA',
    'shots': 1024
}
```

## 📈 Results & Performance

Results are automatically saved to `results/` directory:
- **metrics.json**: Quantitative performance metrics
- **plots/**: Visualization of results
- **comparisons.md**: Detailed analysis

Key metrics evaluated:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, Sharpe Ratio
- Training time, Inference time
- Quantum circuit depth and gate count

## 🌐 Live Demo

The interactive demo provides:
- Real-time market data visualization
- Model prediction comparisons
- Quantum circuit visualization
- Performance monitoring dashboard

Access at: `http://localhost:8501` after running `streamlit run live_demo/app.py`

## 📚 Documentation

Comprehensive documentation available in `docs/`:
- **Literature Review**: Background on QML in finance
- **Methodology**: Detailed approach and algorithms
- **Results**: Experimental findings and analysis
- **Discussion**: Insights and future directions

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-quantum-model`)
3. Commit changes (`git commit -am 'Add new quantum model'`)
4. Push to branch (`git push origin feature/new-quantum-model`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- IBM Qiskit team for quantum computing framework
- Financial data providers
- Quantum machine learning research community

## 📞 Contact

- **Author**: Your Name
- **Email**: your.email@domain.com
- **LinkedIn**: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- **Project Link**: [https://github.com/your-username/HFT-QML-Project](https://github.com/your-username/HFT-QML-Project)

---

## 🔄 Recent Updates

- **v1.2.0**: Added hybrid classical-quantum models
- **v1.1.0**: Implemented real-time prediction system
- **v1.0.0**: Initial release with basic QML models

**Star ⭐ this repository if you find it helpful!**
