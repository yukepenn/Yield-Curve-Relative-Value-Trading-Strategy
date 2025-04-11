# Systematic Testing Error Documentation

## Overview
This document outlines common errors encountered during systematic testing, their causes, and resolution steps. It includes specific sections for LSTM model errors and data preparation issues.

## Error Categories

### 1. Data Loading Errors
- **Missing Data Files**
  - Error: "Expected sequence or array-like, got <class 'NoneType'>"
  - Required Files:
    - `data/processed/features.csv`
    - `data/processed/targets.csv`
    - `results/feature_analysis/y_{spread}_{prediction_type}/selected_features.csv`
  - Resolution: Ensure all required files exist and have proper permissions

### 2. LSTM-Specific Errors
- **Tensor Dimension Mismatch**
  - Error: "Expected 3D input, got [batch_size, sequence_length]"
  - Cause: Incorrect reshaping of input tensors
  - Resolution: Proper tensor reshaping in `_prepare_lstm_data`

- **Data Type Inconsistency**
  - Error: "Expected float32 tensor"
  - Cause: Mixed data types in features/targets
  - Resolution: Explicit type conversion to float32

- **Device Mismatch**
  - Error: "Expected all tensors to be on the same device"
  - Resolution: Proper device handling for CPU/GPU

### 3. Model Training Errors
- **Memory Issues**
  - Error: "CUDA out of memory"
  - Resolution: Batch size adjustment, tensor cleanup

- **Convergence Issues**
  - Error: "Loss is NaN"
  - Resolution: Learning rate adjustment, gradient clipping

## Error Log Structure
```json
{
    "timestamp": "YYYY-MM-DD HH:MM:SS",
    "model_type": "lstm|ridge|lasso|rf|xgb",
    "spread": "2s10s|5s30s|...",
    "prediction_type": "next_day|direction|ternary",
    "error_type": "data_loading|model_training|validation",
    "error_message": "Detailed error message",
    "stack_trace": "Full stack trace",
    "model_params": {
        "batch_size": 32,
        "sequence_length": 20,
        "...": "..."
    },
    "data_shapes": {
        "features": [n_samples, n_features],
        "targets": [n_samples]
    }
}
```

## Debugging Steps
1. **Data Validation**
   - Check file existence
   - Verify data formats
   - Validate feature selection

2. **Model Training**
   - Monitor loss values
   - Check tensor shapes
   - Verify device placement

3. **Error Recovery**
   - Log detailed error information
   - Save model state if possible
   - Clean up resources

## Error Resolution Checklist
- [ ] Verify all required files exist
- [ ] Check data types and shapes
- [ ] Validate model parameters
- [ ] Monitor system resources
- [ ] Review error logs
- [ ] Test fixes in isolation
- [ ] Update documentation

## Next Steps
1. Implement comprehensive error logging
2. Add detailed error messages
3. Create recovery procedures
4. Document successful fixes

## Version History
- 2024-04-10: Added LSTM-specific error handling
- 2024-04-10: Initial documentation creation

## Error Categories

### 1. Data Loading Errors
- **Error Message**: "Expected sequence or array-like, got <class 'NoneType'>"
- **Affected Components**:
  - All spreads (2s10s, 5s30s, 2s5s, 10s30s, 3m10y)
  - All model types (ridge, lasso, rf, xgb)
  - All prediction types (next_day, direction, ternary)
- **Root Cause**: Data loading process failing due to missing or incorrect data files
- **Required Files**:
  - `data/processed/features.csv`
  - `data/processed/targets.csv`
  - `results/feature_analysis/y_{spread}_{prediction_type}/selected_features.csv`

### 2. Data Validation Errors
- **Error Message**: "Data validation failed"
- **Affected Components**:
  - All model types
  - All prediction types
- **Validation Checks**:
  - Missing values
  - Data types
  - Infinite values
  - Sufficient data points
  - Target value ranges

### 3. Model Training Errors
- **Error Message**: "Training failed"
- **Affected Components**:
  - Specific model types
  - Specific prediction types
- **Common Causes**:
  - Hyperparameter issues
  - Memory constraints
  - Data format mismatches

## Error Log Structure

### 1. Error Log File
Location: `results/systematic_testing/{spread}_systematic_test_errors.json`
Format:
```json
{
    "timestamp": "YYYY-MM-DD HH:MM:SS",
    "spread": "spread_name",
    "prediction_type": "prediction_type",
    "model_type": "model_type",
    "error_type": "error_category",
    "error_message": "detailed_error_message",
    "stack_trace": "full_stack_trace",
    "data_info": {
        "num_features": 0,
        "num_samples": 0,
        "date_range": {
            "start": "YYYY-MM-DD",
            "end": "YYYY-MM-DD"
        }
    }
}
```

### 2. Test Results File
Location: `results/systematic_testing/{spread}_systematic_test_results.json`
Format:
```json
{
    "spread": "spread_name",
    "prediction_type": "prediction_type",
    "model_type": "model_type",
    "status": "success/error",
    "metrics": {
        "mse": null,
        "accuracy": null,
        "f1": null,
        "roc_auc": null,
        "train_loss": null,
        "val_loss": null
    },
    "error_info": {
        "error_type": "error_category",
        "error_message": "detailed_error_message"
    }
}
```

## Debugging Steps

### 1. Data Loading Issues
1. Verify file existence:
   ```bash
   ls -l data/processed/features.csv
   ls -l data/processed/targets.csv
   ls -l results/feature_analysis/y_2s10s_next_day/selected_features.csv
   ```

2. Check data format:
   ```python
   import pandas as pd
   features = pd.read_csv('data/processed/features.csv')
   print(features.info())
   print(features.head())
   ```

### 2. Data Validation Issues
1. Run data validation:
   ```python
   from src.model_training import ModelTrainer
   trainer = ModelTrainer(spread='2s10s', prediction_type='next_day', model_type='ridge')
   features, target = trainer.load_data()
   ```

2. Check validation results:
   ```python
   print(features.isnull().sum())
   print(target.isnull().sum())
   print(features.dtypes)
   ```

### 3. Model Training Issues
1. Check model configuration:
   ```python
   print(trainer.model.get_params())
   ```

2. Verify data shapes:
   ```python
   print(features.shape)
   print(target.shape)
   ```

## Error Resolution Checklist

### 1. Data Loading
- [ ] Verify all required files exist
- [ ] Check file permissions
- [ ] Validate data formats
- [ ] Ensure proper feature selection

### 2. Data Validation
- [ ] Handle missing values
- [ ] Convert data types
- [ ] Remove infinite values
- [ ] Ensure sufficient data points

### 3. Model Training
- [ ] Verify hyperparameters
- [ ] Check memory usage
- [ ] Validate data shapes
- [ ] Ensure proper scaling

## Next Steps
1. Implement proper error logging
2. Add detailed error messages
3. Create error recovery procedures
4. Document successful fixes

## Version History
- 2024-04-10: Initial documentation created
- 2024-04-10: Added error categories and debugging steps 