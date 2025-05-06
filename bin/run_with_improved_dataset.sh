#!/bin/bash
# Run the GSPHAR model with the improved dataset implementation

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Get the project root directory (parent of the bin directory)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# Change to the project root directory
cd "$PROJECT_ROOT"

# Run the model with the improved dataset
echo "Running GSPHAR model with improved dataset implementation..."
python scripts/utils/use_improved_dataset.py "$@"

echo "Done!"
