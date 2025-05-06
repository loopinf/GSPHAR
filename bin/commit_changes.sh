#!/bin/bash

# Commit the changes to revert back to the original model
git add src/models/gsphar.py
git commit -m "Revert to original model with torch.complex for better performance"

# Add the run_on_cpu.py script
git add scripts/utils/run_on_cpu.py
git commit -m "Add script to run model on CPU"

echo "Changes committed successfully!"
echo "To train the model on CPU, use: python scripts/train.py --device cpu"
echo "To run a quick test on CPU, use: python scripts/utils/run_on_cpu.py"
