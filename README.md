# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 10:19:27 2024

@author: steve
"""

# Story Points Prediction Tool

This Python script uses Bayesian inference to predict story points for Jira issues based on their summaries and descriptions. It implements both capped (≤8 points) and uncapped models, with support for multiple probability distributions.

## Features

- Automated story point estimation using text features
- Distribution analysis to find best-fitting probability models
- Support for multiple prior distributions (Historical and Jeffreys)
- Capped predictions (≤8 points) to match common Fibonacci-style point systems
- Comprehensive visualization and analysis tools
- Batch inference on new Jira issues

## Prerequisites

The script requires the following Python packages:
```
numpy
pandas
scipy
pymc
matplotlib
sklearn
```

## Input Data Format

The script expects CSV files with the following columns:
- Required for training data: `Summary`, `Description`, `Story Points`
- Required for inference: `Summary`, `Description`

### Example Training Data Format:
```csv
Summary,Description,Story Points
"Add login button","Create a new login button component",2
"Fix database connection","Debug and fix intermittent DB connection issues",5
```

## Usage

1. Prepare your data files:
   - `data/jira_data_with_story_points.csv` (training data)
   - `data/jira_data_without_story_points.csv` (issues needing estimates)

2. Run the script:
```python
python story_points_prediction.py
```

3. The script will:
   - Clean and analyze the training data
   - Train both capped and uncapped models
   - Generate visualizations of model performance
   - Create predictions for new issues
   - Save results to `story_point_inference.csv`

## Key Functions

- `clean_story_points()`: Data cleaning and validation
- `analyze_distributions()`: Fits and compares probability distributions
- `model_with_prior()`: Trains Bayesian model with historical prior
- `model_with_jeffreys_prior()`: Trains model with non-informative prior
- `run_inference()`: Generates predictions for new issues

## Output

The script produces several visualizations:
- Distribution fitting comparisons
- Prediction accuracy plots
- Error analysis by story point size
- Prediction distributions for new issues

The final predictions are saved to `story_point_inference.csv` with predicted story points for each issue.

## Model Details

The script uses:
- TF-IDF vectorization for text features
- Bayesian inference with PyMC
- Multiple probability distributions (Gamma, Log-normal, Beta)
- Historical and Jeffreys priors
- Optional capping at 8 story points

## Performance Metrics

The model's performance is evaluated using:
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- Weighted Mean Absolute Percentage Error (WMAPE)
- Error breakdown by story point size

## Customization

You can modify key parameters in the script:
- Story point cap value (default: 8)
- Number of TF-IDF features (default: 100)
- MCMC sampling parameters
- Input/output file paths

## Contributing

Suggestions for improvements are welcome. Please ensure any modifications maintain the core functionality of:
1. Distribution analysis
2. Model training with multiple priors
3. Capped/uncapped prediction options
4. Comprehensive error analysis

## License

This script is provided under the MIT license. See LICENSE file for details.