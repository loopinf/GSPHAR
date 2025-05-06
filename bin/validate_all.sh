#!/bin/bash
# Run all validation tests

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Get the project root directory (parent of the bin directory)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# Change to the project root directory
cd "$PROJECT_ROOT"

# Run validation
echo "Running validation tests..."
python validate.py

# Run unit tests
echo "Running unit tests..."
pytest tests/

echo "All validation tests completed."
