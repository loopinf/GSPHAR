#!/usr/bin/env python
"""
Cleanup script to remove redundant files while preserving validation functionality.
"""

import os
import sys

def main():
    """
    Remove redundant files while preserving validation functionality.
    """
    print("This script will remove redundant files while preserving validation functionality.")
    print("The following files will be removed:")
    print("  - Redundant utility files in the src directory")
    print("  - Any remaining files in the root directory that have been moved to benchmarks/, scripts/utils/, or bin/")

    # Ask for confirmation
    response = input("Are you sure you want to proceed? (y/n): ")
    if response.lower() != 'y':
        print("Cleanup aborted.")
        sys.exit(0)

    # List of files to remove
    files_to_remove = [
        # These files are redundant with their counterparts in the appropriate subdirectories
        'src/data_utils.py',
        'src/dataset.py',
        'src/models.py',
        'src/utils.py',
        # These test files have been moved to the tests directory
        'test_model.py',
        'test_train.py',
        # These files have been moved to benchmarks/ or scripts/utils/
        'benchmark_devices.py',
        'benchmark_inference.py',
        'benchmark_tensor_ops.py',
        'simple_mps_test.py',
        'test_gsphar_mps.py',
        'compare_models.py',
        'run_on_cpu.py',
        'commit_changes.sh',
    ]

    # Remove files
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            print(f"Removing file: {file_path}")
            os.remove(file_path)
        else:
            print(f"File not found: {file_path}")

    print("\nCleanup completed successfully.")
    print("\nNote: The tmp/ directory has been preserved to maintain validation functionality.")
    print("The validation script (scripts/validation/validate_refactoring.py) depends on files in this directory.")

if __name__ == '__main__':
    main()
