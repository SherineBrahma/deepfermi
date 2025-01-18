#!/bin/bash

# Use Python to get the path to the torch package's lbfgs.py file
original_lbfgs_path=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'optim', 'lbfgs.py'))")
modified_lbfgs_path="${original_lbfgs_path%/*}/lbfgs.py"

echo "Replaced contents in lbfgs.py with that in modified_lbfgs.py..." 
echo "File path: $original_lbfgs_path" 

# Replace old lbfgs file with new file
cp post_install/modified_lbfgs.py $modified_lbfgs_path
