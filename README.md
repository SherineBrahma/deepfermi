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
Create a new environment with Python 3.8 (e.g. using conda) and activate it:
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

## Usage

### Step 1: Generate DCE Perfusion Data
Before proceeding with training or testing, you first need to generate the DCE perfusion data. This can be done by running the ```data_generation.py``` script using the XCAT phantom file provided:

```
python src/deepfermi/data_generation.py
```

This will create a DCE perfusion dataset ```dce_perfusion_data.npz``` in the data folder. You can customize the relevant parameters for generating data in ```data_generation.py```.

Note: Only five cardiac slices are provided in this repository for training, validation, and testing.

### Step 2: Choose Your Option
#### Option 1: Test a Pre-Trained Network
If you would like to test a pre-trained network (without training it yourself), you can directly run the following command:
```
sh script/test_job_queue.sh
```
This script will:

1. Load the pre-trained model.
2. Test the model in two scenarios:
   * With motion artifacts.
   * Without motion artifacts.
3. Generate the required output arrays.

After testing, you can analyze the output arrays in different ways:

  *  Visualize the results by running:
```
 python src/deepfermi/analysis/generate_img.py
```
  *  Evaluate performance metrics by running:
```
 python src/deepfermi/analysis/evaluate_measures.py
```
You can also write your scripts to analyze the arrays.

#### Option 2: Train the Network from Scratch
If you would prefer to train the network from scratch, you can do so after generating the data by running:

```
sh script/train_job_queue.sh
```
After training the model, you can proceed with testing it. However, keep in mind that the training data provided in this repository is quite small and intended primarily for code demonstration purposes, so the model's performance might not be optimal.
