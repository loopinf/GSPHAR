#!/bin/bash
# Generate a comparison plot between the legacy and improved dataset implementations

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Get the project root directory (parent of the bin directory)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# Change to the project root directory
cd "$PROJECT_ROOT"

# Check if psutil and memory_profiler are installed
if ! python -c "import psutil" &> /dev/null; then
    echo "Installing psutil..."
    pip install psutil
fi

if ! python -c "import memory_profiler" &> /dev/null; then
    echo "Installing memory_profiler..."
    pip install memory_profiler
fi

# Run the comparison script
echo "Generating comparison plot..."
python scripts/utils/generate_comparison_plot.py "$@"

# Check if the plot was created
if [ -f "comparison_plot.png" ]; then
    echo "Comparison plot generated successfully!"
    echo "The plot is saved as comparison_plot.png"
    
    # Try to open the plot if on macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Opening the plot..."
        open comparison_plot.png
    fi
else
    echo "Error: Failed to generate comparison plot."
fi
