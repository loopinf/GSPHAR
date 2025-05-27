# Models Directory Organization

This directory contains all trained GSPHAR models, organized by status and experiment type.

## Directory Structure

- **`active/`** - Current best performing models
  - Latest best models ready for use
  - Includes metadata files for each model

- **`experiments/`** - Experimental models organized by type
  - `loss_functions/` - Models trained with different loss functions
  - `two_stage/` - Two-stage training approach models
  - `flexible_gsphar/` - Flexible GSPHAR variant models
  - `crypto_specific/` - Cryptocurrency-specific models

- **`archive/`** - Historical and deprecated models
  - `2025-05-12/` - Models from May 12, 2025
  - `2025-05-13/` - Models from May 13, 2025
  - `2025-05-14/` - Models from May 14, 2025
  - `deprecated/` - Old and unused models

- **`comparison/`** - Model comparison experiments

## Model Naming Convention

Models follow the pattern: `GSPHAR_{parameters}_{dataset}_{loss_type}_{timestamp}.pt`

Best models also have corresponding `*_metadata.json` files with training details.

## Usage

- Use models from `active/` for current work
- Reference `experiments/` for specific experimental approaches
- Archive contains historical models for reference
