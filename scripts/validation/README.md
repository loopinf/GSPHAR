# Validation Scripts

This directory contains scripts for validating the refactored GSPHAR implementation against the original implementation.

## Scripts

- `validate_refactoring.py`: Compares the predictions from the original and refactored implementations.

## Usage

To run the validation script:

```bash
cd /path/to/GSPHAR
python scripts/validation/validate_refactoring.py
```

The script will:

1. Run both the original and refactored implementations
2. Compare the predictions
3. Generate comparison metrics
4. Create visualizations of the results
5. Save the results to the `results/` directory

## Output

The script generates the following outputs:

- `results/comparison_results.csv`: CSV file with comparison metrics for each market index
- `results/comparison_plot.png`: Plot comparing the predictions from both implementations

## Metrics

The script computes the following metrics for each market index:

- Mean Absolute Difference
- Correlation
- Mean Squared Error
- Mean Absolute Percentage Error (%)
