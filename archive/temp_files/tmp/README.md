# Temporary Directory for Validation

This directory contains the original implementation of the GSPHAR model extracted from the Jupyter notebook. These files are used by the validation script (`scripts/validation/validate_refactoring.py`) to compare the original implementation with the refactored version.

## Important Note

**Do not delete this directory** if you want to maintain the validation functionality. The validation script depends on the files in this directory to run the original implementation for comparison purposes.

## Files

- `d_GSPHAR.py`: Original GSPHAR implementation extracted from the Jupyter notebook
- `config.py`: Configuration file for the original implementation
- Other utility files used by the original implementation

## Usage

These files are not meant to be used directly. Instead, they are imported by the validation script to run the original implementation and compare its outputs with the refactored version.

To run the validation:

```bash
python scripts/validation/validate_refactoring.py
```

Or use the convenience script:

```bash
python validate.py
```
