#!/usr/bin/env python
"""
Convenience script to run the validation script from the root directory.
"""

import os
import sys
import subprocess

def main():
    """
    Run the validation script.
    """
    # Get the path to the validation script
    validation_script = os.path.join('scripts', 'validation', 'validate_refactoring.py')
    
    # Check if the script exists
    if not os.path.exists(validation_script):
        print(f"Error: Validation script not found at {validation_script}")
        sys.exit(1)
    
    # Run the script
    print(f"Running validation script: {validation_script}")
    subprocess.run([sys.executable, validation_script])

if __name__ == '__main__':
    main()
