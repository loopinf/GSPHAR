#!/bin/bash
# Run all benchmarks and generate a comprehensive report

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Get the project root directory (parent of the bin directory)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# Change to the project root directory
cd "$PROJECT_ROOT"

# Create results directory if it doesn't exist
mkdir -p results/benchmarks

# Run all benchmarks
echo "Running all benchmarks..."

echo "1. Running inference benchmark..."
python benchmarks/benchmark_inference.py --batch_size 32 --num_runs 50

echo "2. Running tensor operations benchmark..."
python benchmarks/benchmark_tensor_ops.py

echo "3. Running model comparison..."
python scripts/utils/compare_models.py

echo "All benchmarks completed. Results saved to the results/benchmarks directory."
