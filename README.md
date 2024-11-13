# DeepFermi
A deep learning framework for quantifying MR myocardial perfusion

![Static Badge](https://img.shields.io/badge/PyTorch-%23EE4C2C?style=for-the-badge&logo=pytorch&labelColor=black&color=%23EE4C2C)

## Installation

### 1. Clone the repository
Clone the repository to your local machine
```
git clone https://github.com/SherineBrahma/deepfermi.git
```

### 2. Set Up a Python Environment
Create a new Python environment with Python 3.8 (e.g., using conda) and activate it:
``` 
conda create -n deepfermi python=3.8
conda activate deepfermi
```

### 3. Install Dependencies
Install DeepFermi in editable mode along with linting and testing dependencies:
```
pip install -e ".[lint,test]"
```
This command will install DeepFermi and necessary tools for code linting and running tests.

### 4. Setup Pre-Commit Hook
To automatically run pre-commit hooks (like linting) before each commit, install the hooks:
```
pre-commit install
```
