# Archive Directory

This directory contains files that are no longer actively used in the main codebase but are kept for reference or historical purposes.

## Validation Scripts

The `validation` directory contains scripts that were used to validate the original refactoring of the GSPHAR implementation:

- `verify_dataset_implementations.py`: Verifies that the legacy and improved dataset implementations produce the same results.
- `validate_refactoring.py`: Compares the original GSPHAR implementation with the refactored version to ensure they produce similar results.

These scripts are not updated to include the new date-aware dataset approach implemented in `src/utils/date_aware_dataset.py`. For validation of the date-aware dataset approach, refer to the comprehensive tests in `tests/test_date_aware_dataset.py`.

## Why These Files Are Archived

These files are archived because:

1. They are no longer actively used in the main workflow
2. They validate older implementations that have been superseded
3. They provide historical context for the refactoring process
4. They may be useful for reference in future refactoring efforts

If you need to validate the original refactoring again, you can use these scripts, but be aware that they may need to be updated to work with the current codebase.
