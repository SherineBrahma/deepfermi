#!/bin/bash

deepfermi_home_dir=$(dirname "$(dirname "$(realpath "$0")")")

# Replace lbfgs file
# # Use Python to get the path to the torch package's lbfgs.py file
original_lbfgs_path=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'optim', 'lbfgs.py'))")
modified_lbfgs_path="${original_lbfgs_path%/*}/lbfgs.py"

echo "Replaced contents in lbfgs.py with that in modified_lbfgs.py..." 
echo "File path: $original_lbfgs_path" 

# # Replace old lbfgs file with new file
cp post_install/modified_lbfgs.py $modified_lbfgs_path

# Setup pretrained network
# # Create and copy pretrained deepfermi
mkdir -p $deepfermi_home_dir/experiments/01_01_deepfermi_pretrained
cp $deepfermi_home_dir/post_install/deepfermi_pretrained $deepfermi_home_dir/experiments/01_01_deepfermi_pretrained/unet
