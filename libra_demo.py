#!/usr/bin/env python3
"""
Libra Interactive Demo
======================

This interactive demo showcases Libra's automated machine learning capabilities.
Libra automates the end-to-end machine learning process with just one line of code,
making it accessible to both technical and non-technical users.

Features demonstrated:
- Classification tasks
- Regression tasks  
- Neural networks
- Automated data preprocessing
- Model evaluation and visualization

Author: Libra Demo
License: MIT
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from typing import Optional

# Try to import libra, provide helpful error message if not installed
try:
    from libra import client
except ImportError:
    print("❌ Error: Libra is not installed!")
    print("Please install it using: pip install libra")
    print("Then run this demo again.")
    sys.exit(1)


class LibraDemo:
    """Interactive demo class for showcasing Libra's capabilities."""
    
    def __init__(self):
        self.demo_data_dir = "demo_data"
        self.ensure_demo_data_dir()
        
    def ensure_demo_data_dir(self):
        """Create demo data directory if it doesn't exist."""
        if not os.path.exists(self.demo_data_dir):
            os.makedirs(self.demo_data_dir)
    
    def print_header(self, title: str):
        """Print a formatted header for demo sections."""
        print("
        print(f"  {title}")
        print("="*60)
    
    def print_info(self, message: str):
        """Print an info message with formatting."""
        print(f"ℹ️  {message}")
    
    def print_success(self, message: str):
        """Print a success message with formatting."""
        print(f"✅ {message}")
    
    def print_warning(self, message: str):
        """Print a warning message with formatting."""
        print(f"⚠️  {message}")
    
    def wait_for_user(self, message: str = "Press Enter to continue..."):
        """Wait for user input before proceeding."""
        input(f"
    
    def create_sample_classification_data(self) -> str:
        """Create sample data for classification demo."""
        # Create a simple dataset for binary classification
        np.random.seed(42)
        n_samples = 1000
        
        # Generate features
        age = np.random.normal(35, 10, n_samples)
        income = np.random.normal(50000, 15000, n_samples)
        education_years = np.random.normal(14, 3, n_samples)
        
        # Create target variable (loan approval) based on features
        loan_score = (age * 0.1 + income * 0.00001 + education_years * 2 + 
                     np.random.normal(0, 5, n_samples))
        loan_approved = (loan_score > 10).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame({
            'age': age.round(0).astype(int),
            'income': income.round(0).astype(int),
            'education_years': education_years.round(1),
            'loan_approved': loan_approved
        })
        
        # Save to CSV
        filepath = os.path.join(self.demo_data_dir, "loan_data.csv")
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def create_sample_regression_data(self) -> str:
        """Create sample data for regression demo."""
        np.random.seed(42)
        n_samples = 800
        
        # Generate house features
        size_sqft = np.random.normal(2000, 500, n_samples)
        bedrooms = np.random.poisson(3, n_samples)
        bathrooms = np.random.normal(2, 0.5, n_samples)
        age_years = np.random.uniform(0, 50, n_samples)
        
        # Create target variable (house price) based on features
        price = (size_sqft * 150 + bedrooms * 10000 + bathrooms * 15000 - 
                age_years * 1000 + np.random.normal(0, 20000, n_samples))
        
        # Ensure positive prices
        price = np.maximum(price, 50000)
        
        # Create DataFrame
        df = pd.DataFrame({
            'size_sqft': size_sqft.round(0).astype(int),
            'bedrooms': bedrooms,
            'bathrooms': bathrooms.round(1),
            'age_years': age_years.round(0).astype(int),
            'price': price.round(0).astype(int)
        })
        
        # Save to CSV
        filepath = os.path.join(self.demo_data_dir, "house_prices.csv")
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def create_sample_neural_network_data(self) -> str:
        """Create sample data for neural network demo."""
        np.random.seed(42)
        n_samples = 1200
        
        # Generate customer features for churn prediction
        monthly_charges = np.random.normal(70, 20, n_samples)
        total_charges = np.random.normal(2000, 1000, n_samples)
        tenure_months = np.random.poisson(24, n_samples)
        contract_length = np.random.choice([1, 12, 24], n_samples, p=[0.3, 0.4, 0.3])
        
        # Create target variable (customer churn) based on features
        churn_score = (-monthly_charges * 0.01 - total_charges * 0.0001 - 
                      tenure_months * 0.05 + contract_length * 0.1 + 
                      np.random.normal(0, 1, n_samples))
        customer_churn = (churn_score > -1).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame({
            'monthly_charges': monthly_charges.round(2),
            'total_charges': total_charges.round(2),
            'tenure_months': tenure_months,
            'contract_length': contract_length,
            'customer_churn': customer_churn
        })
        
        # Save to CSV
        filepath = os.path.join(self.demo_data_dir, "customer_churn.csv")
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def run_classification_demo(self):
        """Run the classification demo."""
        self.print_header("CLASSIFICATION DEMO: Loan Approval Prediction")
        
        self.print_info("Creating sample loan approval dataset...")
        data_file = self.create_sample_classification_data()
        
        # Load and display sample data
        df = pd.read_csv(data_file)
        print(f"
        print(f"   • {len(df)} loan applications")
        print(f"   • Features: age, income, education_years")
        print(f"   • Target: loan_approved (0=rejected, 1=approved)")
        print(f"
        print(df.head())
        
        self.wait_for_user("Ready to train a classification model? Press Enter...")
        
        try:
            self.print_info("Training classification model with Libra...")
            print("🤖 Running: client.classification(data_file, target='loan_approved')")
            
            # This is the magic of Libra - one line of code!
            model = client.classification(data_file, target='loan_approved')
            
            self.print_success("Classification model trained successfully!")
            self.print_info("Libra automatically handled:")
            print("   • Data preprocessing and cleaning")
            print("   • Feature engineering")
            print("   • Model selection and hyperparameter tuning")
            print("   • Cross-validation and evaluation")
            print("   • Visualization generation")
            
        except Exception as e:
            self.print_warning(f"Demo error: {str(e)}")
            self.print_info("This might happen if Libra dependencies are missing or data format issues.")
    
    def run_regression_demo(self):
        """Run the regression demo."""
        self.print_header("REGRESSION DEMO: House Price Prediction")
        
        self.print_info("Creating sample house price dataset...")
        data_file = self.create_sample_regression_data()
        
        # Load and display sample data
        df = pd.read_csv(data_file)
        print(f"
        print(f"   • {len(df)} house records")
        print(f"   • Features: size_sqft, bedrooms, bathrooms, age_years")
        print(f"   • Target: price (in dollars)")
        print(f"
        print(df.head())
        
        self.wait_for_user("Ready to train a regression model? Press Enter...")
        
        try:
            self.print_info("Training regression model with Libra...")
            print("🤖 Running: client.regression(data_file, target='price')")
            
            # Another one-liner with Libra!
            model = client.regression(data_file, target='price')
            
            self.print_success("Regression model trained successfully!")
            self.print_info("Libra automatically handled:")
            print("   • Data normalization and scaling")
            print("   • Feature selection")
            print("   • Multiple algorithm comparison")
            print("   • Performance metrics calculation")
            print("   • Prediction visualization")
            
        except Exception as e:
            self.print_warning(f"Demo error: {str(e)}")
            self.print_info("This might happen if Libra dependencies are missing or data format issues.")
    
    def run_neural_network_demo(self):
        """Run the neural network demo."""
        self.print_header("NEURAL NETWORK DEMO: Customer Churn Prediction")
        
        self.print_info("Creating sample customer churn dataset...")
        data_file = self.create_sample_neural_network_data()
        
        # Load and display sample data
        df = pd.read_csv(data_file)
        print(f"
        print(f"   • {len(df)} customer records")
        print(f"   • Features: monthly_charges, total_charges, tenure_months, contract_length")
        print(f"   • Target: customer_churn (0=stayed, 1=churned)")
        print(f"
        print(df.head())
        
        self.wait_for_user("Ready to train a neural network? Press Enter...")
        
        try:
            self.print_info("Training neural network with Libra...")
            print("🤖 Running: client.neural_network(data_file, target='customer_churn')")
            
            # Neural networks made simple with Libra!
            model = client.neural_network(data_file, target='customer_churn')
            
            self.print_success("Neural network trained successfully!")
            self.print_info("Libra automatically handled:")
            print("   • Network architecture design")
            print("   • Activation function selection")
            print("   • Optimizer and learning rate tuning")
            print("   • Training and validation splitting")
            print("   • Early stopping and regularization")
            
        except Exception as e:
            self.print_warning(f"Demo error: {str(e)}")
            self.print_info("This might happen if Libra dependencies are missing or data format issues.")
    
    def show_welcome_message(self):
        """Display welcome message and introduction."""
        print("
        print("   WELCOME TO THE LIBRA INTERACTIVE DEMO")
        print("🚀" + "="*58 + "🚀")
        
        print("
        print("   Libra automates the entire machine learning pipeline")
        print("   with just ONE LINE OF CODE! Perfect for:")
        print("   • Beginners learning machine learning")
        print("   • Experienced developers who want rapid prototyping")
        print("   • Teams with mixed technical backgrounds")
        
        print("
        print("   1. Classification: Predicting loan approvals")
        print("   2. Regression: Predicting house prices")
        print("   3. Neural Networks: Predicting customer churn")
        
        print("
        print("   ✅ No need for data preprocessing")
        print("   ✅ Automatic model selection")
        print("   ✅ Built-in hyperparameter tuning")
        print("   ✅ Automatic visualization generation")
        print("   ✅ Works with any skill level")
        
        self.wait_for_user("Ready to start? Press Enter to begin the demo...")
    
    def show_conclusion(self):
        """Display conclusion and next steps."""
        self.print_header("DEMO COMPLETE - WHAT'S NEXT?")
        
        print("🎉 Congratulations! You've seen how Libra makes machine learning accessible!")
        
        print("
        print("   • Documentation: https://libradocs.github.io/")
        print("   • GitHub: https://github.com/Palashio/libra")
        print("   • Install: pip install libra")
        
        print("
        print("   • Use your own CSV files")
        print("   • Experiment with different targets")
        print("   • Explore Libra's advanced features")
        
        print("
        print("   With Libra, you can focus on solving problems rather than")
        print("   worrying about implementation details.")
        
        print("
        print("   Thank you for trying the Libra Demo!")
        print("="*60)
    
    def run_demo(self):
        """Run the complete interactive demo."""
        try:
            self.show_welcome_message()
            
            # Run all demo sections
            self.run_classification_demo()
            self.run_regression_demo()
            self.run_neural_network_demo()
            
            self.show_conclusion()
            
        except KeyboardInterrupt:
            print("
        except Exception as e:
            print(f"
            print("Please check your Libra installation and try again.")


def main():
    """Main function to run the demo."""
    demo = LibraDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()

