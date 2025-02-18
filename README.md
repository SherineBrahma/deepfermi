# DeepFermi
A deep learning framework for quantifying MR myocardial perfusion

![Static Badge](https://img.shields.io/badge/PyTorch-%23EE4C2C?style=for-the-badge&logo=pytorch&labelColor=black&color=%23EE4C2C)

https://github.com/user-attachments/assets/0f04b6d3-ca0a-4f88-a6e6-d98361f9ed24


# Installation

## 1. Clone the repository
Clone the repository and create a new Python environment with Python 3.8 (e.g. using conda):
```
git clone https://github.com/SherineBrahma/deepfermi.git
conda create -n deepfermi python=3.8
conda activate deepfermi
```

## 2. Install DeepFermi and dependencies
Install DeepFermi in editable mode along with necessary tools for linting, testing, and post-install setup:
``` 
pip install -e ".[lint,test]"
sh post_install/post_install.sh
```

# Usage

## Simulated DCE Perfusion Data Generation
Before proceeding with training or testing, you first need to generate the DCE perfusion data. This can be done by running the ```data_generation.py``` script using the XCAT phantom file provided:

```
python src/deepfermi/data_generation.py
```

This will create a DCE perfusion dataset ```dce_perfusion_data.npz``` in the data folder. You can customize the relevant parameters for generating data in ```data_generation.py```.

<div align="center">
  <img src="media/simulation_dataset.gif" width="700" height="auto">
</div>

Note: Only five cardiac slices are provided in this repository for training, validation, and testing.

## Pre-Trained Model
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
![Results Image](media/results.png)

  *  Evaluate performance metrics by running:
```
 python src/deepfermi/analysis/evaluate_measures.py
```
You can also write your scripts to analyze the arrays.

## Train DeepFermi from Scratch
If you would prefer to train the network from scratch, you can do so after generating the data by running:

```
sh script/train_job_queue.sh
```
After training the model, you can proceed with testing it. However, keep in mind that the training data provided in this repository is quite small and intended primarily for code demonstration purposes, so the model's performance might not be optimal.

# Contribution

## How to Contribute
1. Fork the repo and create a new branch.
2. Make changes and test them locally.
3. Submit a pull request with your changes.

## Pre-commit Hooks (Optional)
To automatically run linting before each commit, run:

```
pip install pre-commit
pre-commit install
```

## Running Tests
Run tests before submitting a pull request:
```
pytest  
```

## Issues and Feedback
If you encounter any issues, feel free to open an issue on GitHub.
