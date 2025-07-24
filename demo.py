#!/usr/bin/env python3
"""
Interactive Libra Machine Learning Demo

This script demonstrates Libra's automated machine learning capabilities
across different task types: classification, regression, and neural networks.

Libra automates the end-to-end machine learning process with just one line of code,
making it accessible for users of all technical levels.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression, load_iris, load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Try to import libra, provide helpful error message if not installed
try:
    from libra import client
except ImportError:
    print("❌ Libra is not installed!")
    print("Please install it using: pip install libra")
    print("Then run this demo again.")
    sys.exit(1)


class LibraDemo:
    """Interactive demo class for showcasing Libra's capabilities"""
    
    def __init__(self):
        self.demo_data_dir = "demo_data"
        self.ensure_demo_directory()
        
    def ensure_demo_directory(self):
        """Create demo data directory if it doesn't exist"""
        if not os.path.exists(self.demo_data_dir):
            os.makedirs(self.demo_data_dir)
    
    def print_header(self, title):
        """Print a formatted header for demo sections"""
        print("
        print(f"🚀 {title}")
        print("="*60)
    
    def print_step(self, step_num, description):
        """Print a formatted step description"""
        print(f"
        print("-" * 50)
    
    def wait_for_user(self, message="Press Enter to continue..."):
        """Wait for user input to proceed"""
        input(f"
    
    def create_sample_datasets(self):
        """Create and save sample datasets for the demo"""
        self.print_header("Creating Sample Datasets")
        
        # Classification dataset
        print("📊 Creating classification dataset...")
        X_class, y_class = make_classification(
            n_samples=1000, n_features=10, n_informative=5,
            n_redundant=2, n_clusters_per_class=1, random_state=42
        )
        
        # Create feature names
        feature_names = [f'feature_{i+1}' for i in range(X_class.shape[1])]
        
        # Create DataFrame
        class_df = pd.DataFrame(X_class, columns=feature_names)
        class_df['target'] = y_class
        class_df.to_csv(f"{self.demo_data_dir}/classification_data.csv", index=False)
        
        print(f"✅ Classification dataset saved: {class_df.shape[0]} samples, {class_df.shape[1]-1} features")
        
        # Regression dataset
        print("📊 Creating regression dataset...")
        X_reg, y_reg = make_regression(
            n_samples=1000, n_features=8, noise=0.1, random_state=42
        )
        
        feature_names_reg = [f'feature_{i+1}' for i in range(X_reg.shape[1])]
        reg_df = pd.DataFrame(X_reg, columns=feature_names_reg)
        reg_df['target'] = y_reg
        reg_df.to_csv(f"{self.demo_data_dir}/regression_data.csv", index=False)
        
        print(f"✅ Regression dataset saved: {reg_df.shape[0]} samples, {reg_df.shape[1]-1} features")
        
        # Use Iris dataset for neural network demo
        print("📊 Preparing Iris dataset for neural network demo...")
        iris = load_iris()
        iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
        iris_df['species'] = iris.target
        iris_df.to_csv(f"{self.demo_data_dir}/iris_data.csv", index=False)
        
        print(f"✅ Iris dataset saved: {iris_df.shape[0]} samples, {iris_df.shape[1]-1} features")
        
        self.wait_for_user()
    
    def classification_demo(self):
        """Demonstrate Libra's classification capabilities"""
        self.print_header("Classification Demo with Libra")
        
        print("🎯 In this demo, we'll use Libra to build a classification model")
        print("   to predict binary outcomes from our synthetic dataset.")
        print("
        print("   • Data preprocessing and cleaning")
        print("   • Feature selection and engineering")
        print("   • Model selection and hyperparameter tuning")
        print("   • Cross-validation and performance evaluation")
        print("   • Visualization of results")
        
        self.wait_for_user("Ready to start classification? Press Enter...")
        
        self.print_step(1, "Loading the classification dataset")
        data_path = f"{self.demo_data_dir}/classification_data.csv"
        print(f"📁 Loading data from: {data_path}")
        
        # Show data preview
        df = pd.read_csv(data_path)
        print(f"📊 Dataset shape: {df.shape}")
        print("
        print(df.head())
        print(f"
        print(df['target'].value_counts())
        
        self.wait_for_user()
        
        self.print_step(2, "Running Libra classification - The Magic One-Liner!")
        print("🪄 This single line of code will:")
        print("   • Automatically preprocess the data")
        print("   • Try multiple algorithms (Random Forest, SVM, etc.)")
        print("   • Perform hyperparameter optimization")
        print("   • Generate performance metrics and visualizations")
        
        print(f"
        
        self.wait_for_user("Execute the classification? Press Enter...")
        
        try:
            # The magic one-liner!
            results = client.classify(data_path, target='target')
            
            print("
            print("📊 Libra has automatically:")
            print("   ✅ Preprocessed the data")
            print("   ✅ Trained multiple models")
            print("   ✅ Selected the best performing model")
            print("   ✅ Generated performance visualizations")
            print("   ✅ Saved the trained model")
            
            if results:
                print(f"
            
        except Exception as e:
            print(f"⚠️  Error during classification: {str(e)}")
            print("💡 This might be due to Libra version compatibility or data format issues.")
            print("   The demo will continue with other examples.")
        
        self.wait_for_user()
    
    def regression_demo(self):
        """Demonstrate Libra's regression capabilities"""
        self.print_header("Regression Demo with Libra")
        
        print("📈 In this demo, we'll use Libra to build a regression model")
        print("   to predict continuous values from our synthetic dataset.")
        print("
        print("   • Automatic feature scaling and normalization")
        print("   • Multiple algorithm testing (Linear, Random Forest, etc.)")
        print("   • Automated hyperparameter optimization")
        print("   • Performance metrics (R², RMSE, MAE)")
        print("   • Residual plots and prediction visualizations")
        
        self.wait_for_user("Ready to start regression? Press Enter...")
        
        self.print_step(1, "Loading the regression dataset")
        data_path = f"{self.demo_data_dir}/regression_data.csv"
        print(f"📁 Loading data from: {data_path}")
        
        # Show data preview
        df = pd.read_csv(data_path)
        print(f"📊 Dataset shape: {df.shape}")
        print("
        print(df.head())
        print(f"
        print(df['target'].describe())
        
        self.wait_for_user()
        
        self.print_step(2, "Running Libra regression - Another One-Liner!")
        print("🪄 This single command handles everything:")
        print("   • Data preprocessing and feature engineering")
        print("   • Multiple regression algorithm testing")
        print("   • Cross-validation and model selection")
        print("   • Performance evaluation and visualization")
        
        print(f"
        
        self.wait_for_user("Execute the regression? Press Enter...")
        
        try:
            # The magic one-liner for regression!
            results = client.regress(data_path, target='target')
            
            print("
            print("📊 Libra has automatically:")
            print("   ✅ Preprocessed and scaled the data")
            print("   ✅ Tested multiple regression algorithms")
            print("   ✅ Optimized hyperparameters")
            print("   ✅ Generated performance metrics")
            print("   ✅ Created residual plots")
            print("   ✅ Saved the best model")
            
            if results:
                print(f"
            
        except Exception as e:
            print(f"⚠️  Error during regression: {str(e)}")
            print("💡 This might be due to Libra version compatibility or data format issues.")
            print("   The demo will continue with the neural network example.")
        
        self.wait_for_user()
    
    def neural_network_demo(self):
        """Demonstrate Libra's neural network capabilities"""
        self.print_header("Neural Network Demo with Libra")
        
        print("🧠 In this demo, we'll use Libra to build a neural network")
        print("   for multi-class classification using the famous Iris dataset.")
        print("
        print("   • Automatic network architecture selection")
        print("   • Data preprocessing for neural networks")
        print("   • Hyperparameter optimization (layers, neurons, etc.)")
        print("   • Training with early stopping")
        print("   • Performance visualization and model saving")
        
        self.wait_for_user("Ready to build a neural network? Press Enter...")
        
        self.print_step(1, "Loading the Iris dataset")
        data_path = f"{self.demo_data_dir}/iris_data.csv"
        print(f"📁 Loading data from: {data_path}")
        
        # Show data preview
        df = pd.read_csv(data_path)
        print(f"📊 Dataset shape: {df.shape}")
        print("
        print(df.head())
        print(f"
        print(df['species'].value_counts())
        
        self.wait_for_user()
        
        self.print_step(2, "Running Libra neural network - Deep Learning Made Simple!")
        print("🪄 This one line creates a complete neural network:")
        print("   • Automatic architecture design")
        print("   • Data preprocessing for deep learning")
        print("   • Training with optimal parameters")
        print("   • Performance evaluation")
        
        print(f"
        
        self.wait_for_user("Build the neural network? Press Enter...")
        
        try:
            # The magic one-liner for neural networks!
            results = client.neural_network(data_path, target='species')
            
            print("
            print("🧠 Libra has automatically:")
            print("   ✅ Designed the network architecture")
            print("   ✅ Preprocessed data for deep learning")
            print("   ✅ Trained the neural network")
            print("   ✅ Optimized hyperparameters")
            print("   ✅ Generated training visualizations")
            print("   ✅ Saved the trained model")
            
            if results:
                print(f"
            
        except Exception as e:
            print(f"⚠️  Error during neural network training: {str(e)}")
            print("💡 This might be due to Libra version compatibility or data format issues.")
            print("   Neural networks might require additional dependencies.")
        
        self.wait_for_user()
    
    def show_generated_files(self):
        """Show what files Libra has generated"""
        self.print_header("Generated Files and Outputs")
        
        print("📁 Libra automatically generates several files during the ML process:")
        print("
        
        # List files in current directory
        current_files = [f for f in os.listdir('.') if not f.startswith('.')]
        libra_files = [f for f in current_files if any(keyword in f.lower() 
                      for keyword in ['model', 'plot', 'results', 'libra'])]
        
        if libra_files:
            print("
            for file in libra_files:
                print(f"   📄 {file}")
        else:
            print("
            print("   Files might be in subdirectories or have different naming.")
        
        print("
        print("   • Model files (.pkl, .h5)")
        print("   • Performance plots (.png)")
        print("   • Results summaries (.txt, .json)")
        print("   • Feature importance charts")
        print("   • Confusion matrices")
        print("   • Learning curves")
        
        self.wait_for_user()
    
    def run_demo(self):
        """Run the complete interactive demo"""
        self.print_header("Welcome to the Interactive Libra ML Demo!")
        
        print("🎯 This demo showcases Libra's automated machine learning capabilities.")
        print("   Libra makes machine learning accessible with just one line of code!")
        print("
        print("   1. Create sample datasets")
        print("   2. Classification demo")
        print("   3. Regression demo") 
        print("   4. Neural network demo")
        print("   5. Review generated files")
        
        print("
        print("   • No ML expertise required")
        print("   • Automatic preprocessing")
        print("   • Multiple algorithm testing")
        print("   • Hyperparameter optimization")
        print("   • Automatic visualization")
        
        self.wait_for_user("Ready to start? Press Enter...")
        
        # Run demo sections
        self.create_sample_datasets()
        self.classification_demo()
        self.regression_demo()
        self.neural_network_demo()
        self.show_generated_files()
        
        self.print_header("Demo Complete! 🎉")
        print("🚀 You've seen how Libra automates the entire ML pipeline!")
        print("💡 Next steps:")
        print("   • Try Libra with your own datasets")
        print("   • Explore the generated visualizations")
        print("   • Check out the Jupyter notebook version")
        print("   • Read the documentation for advanced features")
        print("


if __name__ == "__main__":
    demo = LibraDemo()
    demo.run_demo()

