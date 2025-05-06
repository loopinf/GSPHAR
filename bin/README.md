# Executable Scripts

This directory contains executable scripts that can be run directly from the command line. These scripts provide convenient shortcuts for common tasks.

## Scripts

- `run_gsphar.sh`: Runs the GSPHAR model with default settings
- `benchmark_all.sh`: Runs all benchmarks and generates a comprehensive report
- `validate_all.sh`: Runs all validation tests

## Usage

To run a script:

```bash
# Make sure the script is executable
chmod +x bin/run_gsphar.sh

# Run the script
./bin/run_gsphar.sh
```

You can also add the `bin` directory to your PATH to run the scripts from anywhere:

```bash
export PATH=$PATH:/path/to/GSPHAR/bin
```

Then you can run the scripts directly:

```bash
run_gsphar.sh
```
