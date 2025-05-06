# Benchmarks

This directory contains benchmark scripts for evaluating the performance of the GSPHAR model on different hardware and with different configurations.

## Scripts

- `benchmark_devices.py`: Compares model performance across different devices (CPU, MPS, CUDA)
- `benchmark_inference.py`: Benchmarks inference performance
- `benchmark_tensor_ops.py`: Benchmarks tensor operations performance
- `simple_mps_test.py`: Simple test for MPS (Metal Performance Shaders) compatibility
- `test_gsphar_mps.py`: Tests GSPHAR model on MPS

## Usage

To run a benchmark script:

```bash
python benchmarks/benchmark_inference.py
```

The benchmark scripts will output performance metrics and may generate visualizations in the `results/` directory.
