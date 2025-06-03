#!/usr/bin/env python3
"""
Human Activity Recognition (HAR) Analysis
This script implements a comprehensive HAR system using the HCA Dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import time
import warnings
import os
import joblib
warnings.filterwarnings('ignore')

class HARAnalysis:
    def __init__(self):
        """Initialize the HAR analysis class."""
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42)
        }
        self.results = {}
        self.activity_labels = {
            1: 'WALKING',
            2: 'WALKING_UPSTAIRS',
            3: 'WALKING_DOWNSTAIRS',
            4: 'SITTING',
            5: 'STANDING',
            6: 'LAYING'
        }
        
        # Create results directory if it doesn't exist
        self.results_dir = 'model_results'
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def load_data(self):
        """
        Load and preprocess the HAR dataset from the HAR Dataset folder.
        """
        print("Loading and preprocessing data...")
        
        # Load training data
        self.X_train = np.loadtxt('HAR Dataset/train/X_train_data.txt')
        self.y_train = np.loadtxt('HAR Dataset/train/y_train_data.txt')
        
        # Load test data
        self.X_test = np.loadtxt('HAR Dataset/test/X_test_data.txt')
        self.y_test = np.loadtxt('HAR Dataset/test/y_test_data.txt')
        
        # Adjust labels to start from 0 (subtract 1 from all labels)
        self.y_train = self.y_train - 1
        self.y_test = self.y_test - 1
        
        # Scale the features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Data loaded successfully. Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print("\nActivity Labels:")
        for label_id, label_name in self.activity_labels.items():
            print(f"{label_id-1}: {label_name}")

    def save_model_results(self, model_name, model, y_pred, metrics):
        """
        Save individual model results to files.
        
        Args:
            model_name (str): Name of the model
            model: Trained model object
            y_pred: Model predictions
            metrics (dict): Dictionary containing model metrics
        """
        # Create model-specific directory
        model_dir = os.path.join(self.results_dir, model_name.replace(' ', '_'))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Save the model
        joblib.dump(model, os.path.join(model_dir, f'{model_name.replace(" ", "_")}_model.joblib'))
        
        # Save predictions
        np.savetxt(os.path.join(model_dir, 'predictions.txt'), y_pred)
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(model_dir, 'metrics.csv'), index=False)
        
        # Save detailed classification report
        report = classification_report(self.y_test, y_pred, 
                                     target_names=list(self.activity_labels.values()),
                                     output_dict=True)
        pd.DataFrame(report).transpose().to_csv(os.path.join(model_dir, 'classification_report.csv'))
        
        # Save confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        np.savetxt(os.path.join(model_dir, 'confusion_matrix.txt'), cm)
        
        # Plot and save confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(self.activity_labels.values()),
                   yticklabels=list(self.activity_labels.values()))
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'))
        plt.close()

    def train_and_evaluate(self):
        """Train and evaluate all models."""
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            start_time = time.time()
            
            # Train the model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            # Store results
            metrics = {
                'accuracy': accuracy,
                'f1_score': f1,
                'training_time': time.time() - start_time
            }
            
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1,
                'predictions': y_pred,
                'training_time': metrics['training_time']
            }
            
            # Save individual model results
            self.save_model_results(name, model, y_pred, metrics)
            
            print(f"{name} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Training Time: {metrics['training_time']:.2f} seconds")
            
            # Print detailed classification report
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred, target_names=list(self.activity_labels.values())))

    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models."""
        n_models = len(self.models)
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        axes = axes.ravel()
        
        for idx, (name, result) in enumerate(self.results.items()):
            cm = confusion_matrix(self.y_test, result['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=list(self.activity_labels.values()),
                       yticklabels=list(self.activity_labels.values()))
            axes[idx].set_title(f'Confusion Matrix - {name}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('True')
            plt.setp(axes[idx].get_xticklabels(), rotation=45, ha='right')
            plt.setp(axes[idx].get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'all_confusion_matrices.png'))
        plt.close()

    def compare_models(self):
        """Create a comparison table of model performances."""
        comparison = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [result['accuracy'] for result in self.results.values()],
            'F1 Score': [result['f1_score'] for result in self.results.values()],
            'Training Time (s)': [result['training_time'] for result in self.results.values()]
        })
        
        print("\nModel Comparison:")
        print(comparison.to_string(index=False))
        
        # Save comparison to CSV
        comparison.to_csv(os.path.join(self.results_dir, 'model_comparison.csv'), index=False)

def main():
    """Main function to run the HAR analysis."""
    # Initialize the analysis
    har = HARAnalysis()
    
    # Load the data
    har.load_data()
    
    # Train and evaluate models
    har.train_and_evaluate()
    
    # Plot confusion matrices
    har.plot_confusion_matrices()
    
    # Compare models
    har.compare_models()

if __name__ == "__main__":
    main() 