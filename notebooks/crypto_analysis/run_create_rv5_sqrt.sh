#!/bin/bash
# Run the Jupyter notebook to create rv5_sqrt_38_crypto.csv

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Get the project root directory (parent of the notebooks/crypto_analysis directory)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

# Change to the project root directory
cd "$PROJECT_ROOT"

# Run the notebook
echo "Running Jupyter notebook to create rv5_sqrt_38_crypto.csv..."
jupyter nbconvert --to notebook --execute notebooks/crypto_analysis/create_rv5_sqrt.ipynb --output notebooks/crypto_analysis/create_rv5_sqrt_executed.ipynb

# Check if the output file was created
if [ -f "data/rv5_sqrt_38_crypto.csv" ]; then
    echo "Successfully created data/rv5_sqrt_38_crypto.csv"
    echo "Executed notebook saved as notebooks/crypto_analysis/create_rv5_sqrt_executed.ipynb"
else
    echo "Error: Failed to create data/rv5_sqrt_38_crypto.csv"
fi
