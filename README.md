# DeepFermi
A self-supervised deep learning framework that integrates the Fermi model for fast, accurate, robust, and data-consistent myocardial quantification. For more detailed information, please refer to our publication, 'Robust Myocardial Perfusion MRI Quantification with DeepFermi,' which outlines the methodology and validation of the DeepFermi framework.

<a id="publication"></a>
[Publication](https://ieeexplore.ieee.org/document/10731565) | [Citation](#bibtex-citation)

https://github.com/user-attachments/assets/7e0e10cf-20e7-43e0-8725-bfdbba3d23e3

**Contribution**: Sherine Brahma, Andreas Kofler, Felix F. Zimmermann, Tobias Schaeffter, Amedeo Chiribiri, and Christoph Kolbitsch.

## Network Architecture

<div align="center">
  <img src="media/network_architecture.png" width="600" height="auto">
</div>

# Installation

### 1. Clone the repository
Clone the repository and create a new Python environment with Python 3.8 (e.g. using conda):
```bash
git clone https://github.com/SherineBrahma/deepfermi.git
conda create -n deepfermi python=3.8
conda activate deepfermi
```

### 2. Install DeepFermi and dependencies
Install DeepFermi in editable mode along with necessary tools for linting, testing. Run the post-install setup to configure the environment for testing and training:
```bash 
pip install -e ".[lint,test]"
sh post_install/post_install.sh
```

# Usage

## 1. Simulate DCE Perfusion Data

This repository includes a small dataset of five cardiac slices based on the [XCAT](https://aapm.onlinelibrary.wiley.com/doi/10.1118/1.3480985) phantom, for the purpose of code demonstration. Run the ```data_generation.py``` script to generate myocardial perfusion maps, and to simulate dynamic contrast agent (DCE) MRI images. Furthermore, you can customize the parameters directly in the data_generation.py script.

```python
python src/deepfermi/data_generation.py
```

A DCE perfusion dataset, ```dce_perfusion_data.npz```, in created in the ```data``` folder. The DCE images that are synthesized are further induced with motion outliers to model practical scenarios, as shown in the gif below.

<div align="center">
  <img src="media/simulation_dataset.gif" width="700" height="auto">
</div>

## 2. Pre-Trained Model

A pre-trained model, trained on a larger dataset (see [publication](#bibtex-citation) for details), is included for quick testing. You can directly run the provided testing script to evaluate the pre-trained model on two datasets: i) with motion artifacts, and ii) without motion artifacts.

```python
sh script/test_job_queue.sh
```

The results will be saved as output arrays in the ```experiments``` folder, which can be further assessed. For example, to visualize the generated arrays, run:

```python
 python src/deepfermi/analysis/generate_img.py
```

The example below shows that DeepFermi estimates are more robust to motion artifacts compared to traditional Fermi-deconvolution, which relies on well-established optimization algorithms without deep learning components, such as the  [Limited memory Broyden-Fletcher-Goldfarb-Shanno](https://link.springer.com/article/10.1007/BF01589116) (LBFGS) algorithm.

<div align="center">
  <img src="media/results.png" width="700" height="auto">
</div>  

You can also write custom scripts to analyze the arrays. Additionally, an evaluate_measures.py script is provided for quantitatively assessing the performance of the model.

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

# Citation

<a id="bibtex-citation"></a>

If you would like to cite this work, here is the BibTeX entry:

```bibtex
@article{brahma2024robust,
  title={Robust Myocardial Perfusion MRI Quantification with DeepFermi},
  author={Brahma, Sherine and Kofler, Andreas and Zimmermann, Felix F and Schaeffter, Tobias and Chiribiri, Amedeo and Kolbitsch, Christoph},
  journal={IEEE Transactions on Biomedical Engineering},
  year={2024},
  publisher={IEEE}
}
```
