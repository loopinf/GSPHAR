#!/bin/bash
# Run the simple comparison Jupyter notebook

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Get the project root directory (parent of the notebooks/crypto_analysis directory)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

# Change to the project root directory
cd "$PROJECT_ROOT"

# Run the notebook
echo "Running simple comparison notebook..."
jupyter nbconvert --to notebook --execute notebooks/crypto_analysis/simple_rv5_rtn_comparison.ipynb --output notebooks/crypto_analysis/simple_rv5_rtn_comparison_executed.ipynb

# Check if the executed notebook was created
if [ -f "notebooks/crypto_analysis/simple_rv5_rtn_comparison_executed.ipynb" ]; then
    echo "Successfully executed the simple comparison notebook"
    echo "Executed notebook saved as notebooks/crypto_analysis/simple_rv5_rtn_comparison_executed.ipynb"
else
    echo "Error: Failed to execute the simple comparison notebook"
fi
