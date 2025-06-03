# Human Activity Recognition (HAR) Analysis

This project implements a Human Activity Recognition system using the HCA Dataset. The system classifies six different human activities using inertial sensor data.

## Project Structure
```
HAR Project/
│
├── har_analysis.py          # Main script containing the HAR implementation
├── requirements.txt         # Required Python packages
├── README.md               # Project documentation
│
├── HAR Dataset/            # Dataset directory
│   ├── train/             # Training data
│   │   ├── X_train_data.txt
│   │   ├── y_train_data.txt
│   │   └── subject_train_data.txt
│   │
│   ├── test/              # Test data
│   │   ├── X_test_data.txt
│   │   ├── y_test_data.txt
│   │   └── subject_test_data.txt
│   │
│   ├── features_data.txt          # Feature descriptions
│   ├── features_info_data.txt     # Detailed feature information
│   └── activity_label.txt         # Activity labels
│
└── model_results/          # Directory for model outputs
    ├── Random_Forest/     # Random Forest model results
    ├── SVM/               # SVM model results
    ├── Neural_Network/    # Neural Network model results
    └── XGBoost/          # XGBoost model results
```

## Features
- Data preprocessing and feature extraction
- Implementation of multiple ML models:
  - Random Forest (Baseline)
  - Support Vector Machine (SVM)
  - Neural Network
  - XGBoost
- Comprehensive model evaluation with:
  - Confusion matrices
  - Accuracy metrics
  - F1-scores
- Analysis of real-time deployment possibilities

## Setup
1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the analysis:
```bash
python har_analysis.py
```

## Dataset
The HCA Dataset contains inertial sensor data for six activities:
- Walking
- Sitting
- Standing
- Laying
- Walking Upstairs
- Walking Downstairs

Each sample is represented by a 561-dimensional feature vector.

## Model Performance
The script evaluates and compares the performance of multiple models, providing:
- Classification accuracy
- F1-scores
- Confusion matrices
- Training and inference time analysis

## File Descriptions

### Main Scripts
- `har_analysis.py`: Main implementation file containing the HAR system
  - Data loading and preprocessing
  - Model training and evaluation
  - Performance visualization
  - Results comparison

### Dataset Files
- `X_train_data.txt`: Training feature vectors
- `y_train_data.txt`: Training labels
- `X_test_data.txt`: Test feature vectors
- `y_test_data.txt`: Test labels
- `features_data.txt`: List of all features
- `features_info_data.txt`: Detailed feature descriptions
- `activity_label.txt`: Activity class labels

### Output Files
- Model results are saved in the `model_results` directory
- Each model has its own subdirectory containing:
  - Trained model
  - Predictions
  - Performance metrics
  - Confusion matrix visualization

## Real-time Deployment Considerations
The README includes a section discussing how the HAR pipeline could be adapted for real-time or on-device use, including:
- Model optimization techniques
- Feature extraction optimization
- Deployment considerations 