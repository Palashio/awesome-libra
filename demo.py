#!/usr/bin/env python3
"""
Libra Interactive Demo

This script demonstrates Libra's automated machine learning capabilities
including classification, regression, and neural network creation.

Libra enables machine learning in just one line of code!
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from typing import Optional

# Try to import libra, provide helpful error if not available
try:
    from libra import client
except ImportError:
    print("❌ Libra is not installed!")
    print("Please install it using: pip install libra")
    print("Or run: pip install -r requirements.txt")
    sys.exit(1)


class LibraDemo:
    """Interactive demo class for showcasing Libra's capabilities."""
    
    def __init__(self):
        self.datasets_dir = "demo_datasets"
        self.results_dir = "demo_results"
        self.ensure_directories()
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.datasets_dir, self.results_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def print_header(self, title: str):
        """Print a formatted header."""
        print("\n" + "="*60)
        print(f"🚀 {title}")
        print("="*60)
    
    def print_step(self, step: str):
        """Print a formatted step."""
        print(f"\n📋 {step}")
        print("-" * 40)
    
    def wait_for_user(self, message: str = "Press Enter to continue..."):
        """Wait for user input."""
        input(f"\n⏸️  {message}")
    
    def display_welcome(self):
        """Display welcome message and introduction."""
        self.print_header("Welcome to Libra Interactive Demo!")
        print("""
🎯 What is Libra?
Libra is a Python library that automates machine learning processes
in just a few lines of code. It enables:

• 🤖 Automated Classification
• 📈 Automated Regression  
• 🧠 Neural Network Creation
• 🔍 Automatic Model Selection
• 📊 Built-in Data Preprocessing

✨ Key Features:
- One-liner machine learning models
- Automatic hyperparameter tuning
- Built-in data visualization
- No deep ML expertise required
        """)
        self.wait_for_user()
    
    def show_menu(self) -> str:
        """Display main menu and get user choice."""
        self.print_header("Choose a Demo")
        print("""
1. 🎯 Classification Demo (Predict categories)
2. 📈 Regression Demo (Predict continuous values)
3. 🧠 Neural Network Demo (Deep learning)
4. 📊 Compare All Models
5. 🔍 Custom Dataset Demo
6. ❌ Exit
        """)
        
        while True:
            choice = input("Enter your choice (1-6): ").strip()
            if choice in ['1', '2', '3', '4', '5', '6']:
                return choice
            print("❌ Invalid choice. Please enter 1-6.")
    
    def create_sample_classification_data(self) -> str:
        """Create sample classification dataset."""
        filepath = os.path.join(self.datasets_dir, "classification_data.csv")
        
        if not os.path.exists(filepath):
            print("📊 Creating sample classification dataset...")
            np.random.seed(42)
            
            # Generate synthetic customer data for churn prediction
            n_samples = 1000
            data = {
                'age': np.random.randint(18, 80, n_samples),
                'income': np.random.normal(50000, 20000, n_samples),
                'months_subscribed': np.random.randint(1, 60, n_samples),
                'support_calls': np.random.poisson(2, n_samples),
                'satisfaction_score': np.random.uniform(1, 10, n_samples)
            }
            
            # Create target based on logical rules
            churn_prob = (
                (data['satisfaction_score'] < 5) * 0.4 +
                (data['support_calls'] > 3) * 0.3 +
                (data['months_subscribed'] < 6) * 0.2 +
                np.random.uniform(0, 0.1, n_samples)
            )
            data['will_churn'] = (churn_prob > 0.5).astype(int)
            
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            print(f"✅ Created: {filepath}")
        
        return filepath
    
    def create_sample_regression_data(self) -> str:
        """Create sample regression dataset."""
        filepath = os.path.join(self.datasets_dir, "regression_data.csv")
        
        if not os.path.exists(filepath):
            print("📊 Creating sample regression dataset...")
            np.random.seed(42)
            
            # Generate synthetic house price data
            n_samples = 1000
            data = {
                'bedrooms': np.random.randint(1, 6, n_samples),
                'bathrooms': np.random.randint(1, 4, n_samples),
                'square_feet': np.random.randint(800, 4000, n_samples),
                'age_years': np.random.randint(0, 50, n_samples),
                'garage_spaces': np.random.randint(0, 3, n_samples)
            }
            
            # Create target based on logical rules with noise
            price = (
                data['square_feet'] * 150 +
                data['bedrooms'] * 10000 +
                data['bathrooms'] * 15000 +
                data['garage_spaces'] * 8000 -
                data['age_years'] * 500 +
                np.random.normal(0, 20000, n_samples)
            )
            data['price'] = np.maximum(price, 50000)  # Minimum price
            
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            print(f"✅ Created: {filepath}")
        
        return filepath
    
    def demo_classification(self):
        """Demonstrate Libra's classification capabilities."""
        self.print_header("Classification Demo: Customer Churn Prediction")
        
        print("""
🎯 Scenario: Predict which customers will churn (leave the service)
📊 Dataset: Customer demographics and behavior data
🎯 Target: Binary classification (will_churn: 0 or 1)
        """)
        
        # Create sample data
        filepath = self.create_sample_classification_data()
        
        # Show data preview
        self.print_step("Step 1: Loading and exploring the data")
        df = pd.read_csv(filepath)
        print(f"Dataset shape: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head())
        print(f"\nTarget distribution:")
        print(df['will_churn'].value_counts())
        
        self.wait_for_user()
        
        # Demonstrate Libra's one-liner classification
        self.print_step("Step 2: Creating ML model with Libra (ONE LINE!)")
        print("🚀 Libra Code:")
        print(f"client.classification('{filepath}', target='will_churn')")
        print("\n⏳ Training model... (This may take a moment)")
        
        try:
            # This is the magic one-liner!
            model = client.classification(filepath, target='will_churn')
            print("✅ Model trained successfully!")
            print("📊 Libra automatically:")
            print("   • Preprocessed the data")
            print("   • Selected the best algorithm")
            print("   • Tuned hyperparameters")
            print("   • Evaluated performance")
            print("   • Generated visualizations")
            
        except Exception as e:
            print(f"⚠️  Demo mode - Model training simulated")
            print(f"   In real usage, this would train the model: {e}")
        
        self.wait_for_user()
    
    def demo_regression(self):
        """Demonstrate Libra's regression capabilities."""
        self.print_header("Regression Demo: House Price Prediction")
        
        print("""
📈 Scenario: Predict house prices based on features
🏠 Dataset: House characteristics and prices
🎯 Target: Continuous values (price in dollars)
        """)
        
        # Create sample data
        filepath = self.create_sample_regression_data()
        
        # Show data preview
        self.print_step("Step 1: Loading and exploring the data")
        df = pd.read_csv(filepath)
        print(f"Dataset shape: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head())
        print(f"\nPrice statistics:")
        print(df['price'].describe())
        
        self.wait_for_user()
        
        # Demonstrate Libra's one-liner regression
        self.print_step("Step 2: Creating ML model with Libra (ONE LINE!)")
        print("🚀 Libra Code:")
        print(f"client.regression('{filepath}', target='price')")
        print("\n⏳ Training model... (This may take a moment)")
        
        try:
            # This is the magic one-liner!
            model = client.regression(filepath, target='price')
            print("✅ Model trained successfully!")
            print("📊 Libra automatically:")
            print("   • Preprocessed the data")
            print("   • Selected the best algorithm")
            print("   • Tuned hyperparameters")
            print("   • Evaluated performance")
            print("   • Generated visualizations")
            
        except Exception as e:
            print(f"⚠️  Demo mode - Model training simulated")
            print(f"   In real usage, this would train the model: {e}")
        
        self.wait_for_user()
    
    def demo_neural_network(self):
        """Demonstrate Libra's neural network capabilities."""
        self.print_header("Neural Network Demo: Deep Learning in One Line")
        
        print("""
🧠 Scenario: Create a neural network for classification
⚡ Feature: Deep learning without complexity
🎯 Goal: Show how easy neural networks can be with Libra
        """)
        
        filepath = self.create_sample_classification_data()
        
        self.print_step("Creating Neural Network with Libra")
        print("🚀 Libra Code:")
        print(f"client.neural_network('{filepath}', target='will_churn')")
        print("\n⏳ Building neural network... (This may take a moment)")
        
        try:
            # This is the magic one-liner for neural networks!
            model = client.neural_network(filepath, target='will_churn')
            print("✅ Neural network created successfully!")
            print("🧠 Libra automatically:")
            print("   • Designed the network architecture")
            print("   • Set optimal hyperparameters")
            print("   • Trained the deep learning model")
            print("   • Provided performance metrics")
            
        except Exception as e:
            print(f"⚠️  Demo mode - Neural network training simulated")
            print(f"   In real usage, this would create the neural network: {e}")
        
        self.wait_for_user()
    
    def demo_model_comparison(self):
        """Demonstrate comparing multiple models."""
        self.print_header("Model Comparison Demo")
        
        print("""
🔍 Feature: Automatic model comparison and selection
📊 Libra can test multiple algorithms and pick the best one
⚡ All automated - no manual tuning required!
        """)
        
        filepath = self.create_sample_classification_data()
        
        self.print_step("Comparing Multiple Models")
        print("🚀 Libra automatically tests:")
        print("   • Random Forest")
        print("   • Support Vector Machine")
        print("   • Logistic Regression")
        print("   • Neural Networks")
        print("   • And more...")
        
        print("\n⏳ This would run model comparison...")
