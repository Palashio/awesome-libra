#!/usr/bin/env python3
"""
Libra Interactive Demo

This demo showcases Libra's automated machine learning capabilities including:
- Regression tasks
- Classification tasks
- Neural network tasks
- Automatic data preprocessing
- Model evaluation and visualization

Libra enables machine learning in just one line of code, making it accessible
to both beginners and experienced developers.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from libra import client
    import matplotlib.pyplot as plt
    LIBRA_AVAILABLE = True
except ImportError as e:
    LIBRA_AVAILABLE = False
    print(f"⚠️  Warning: {e}")
    print("Please install required dependencies: pip install -r requirements.txt")


class LibraDemo:
    """Interactive demo class for showcasing Libra's capabilities"""
    
    def __init__(self):
        self.demo_datasets_path = Path("demo_datasets")
        self.results_path = Path("demo_results")
        self.results_path.mkdir(exist_ok=True)
        
    def display_banner(self):
        """Display welcome banner"""
        print("=" * 60)
        print("🚀 LIBRA INTERACTIVE DEMO 🚀")
        print("=" * 60)
        print("Welcome to the Libra automated machine learning demo!")
        print("Libra enables you to build ML models with just one line of code.")
        print("Perfect for beginners and experienced developers alike.\n")
        
        if not LIBRA_AVAILABLE:
            print("❌ Libra is not installed. Please run:")
            print("   pip install -r requirements.txt")
            print("   Then restart this demo.\n")
            return False
        return True
    
    def display_menu(self):
        """Display main menu options"""
        print("\n📋 DEMO OPTIONS:")
        print("1. 📈 Regression Demo (House Price Prediction)")
        print("2. 🎯 Classification Demo (Iris Species Classification)")
        print("3. 🧠 Neural Network Demo (Stock Price Prediction)")
        print("4. 📊 Compare All Models")
        print("5. 🔍 Learn About Libra")
        print("6. 🚪 Exit Demo")
        print("-" * 40)
    
    def get_user_choice(self):
        """Get user menu choice with validation"""
        while True:
            try:
                choice = input("Enter your choice (1-6): ").strip()
                if choice in ['1', '2', '3', '4', '5', '6']:
                    return int(choice)
                else:
                    print("❌ Invalid choice. Please enter a number between 1-6.")
            except KeyboardInterrupt:
                print("\n\n👋 Demo interrupted. Goodbye!")
                sys.exit(0)
            except Exception:
                print("❌ Invalid input. Please enter a number between 1-6.")
    
    def check_dataset(self, filename):
        """Check if dataset exists"""
        filepath = self.demo_datasets_path / filename
        if not filepath.exists():
            print(f"❌ Dataset not found: {filepath}")
            print("Please ensure demo datasets are available in the demo_datasets/ directory.")
            return False
        return True
    
    def regression_demo(self):
        """Demonstrate regression with house price prediction"""
        print("\n" + "="*50)
        print("📈 REGRESSION DEMO - HOUSE PRICE PREDICTION")
        print("="*50)
        
        dataset_file = "house_prices.csv"
        if not self.check_dataset(dataset_file):
            return
        
        print("🏠 This demo predicts house prices based on features like:")
        print("   • Square footage")
        print("   • Number of bedrooms/bathrooms")
        print("   • Location factors")
        print("   • Age of the house")
        
        input("\n📊 Press Enter to load the dataset and start training...")
        
        try:
            # Load and display dataset info
            data_path = str(self.demo_datasets_path / dataset_file)
            df = pd.read_csv(data_path)
            
            print(f"\n📋 Dataset loaded successfully!")
            print(f"   • Rows: {len(df)}")
            print(f"   • Columns: {len(df.columns)}")
            print(f"   • Features: {', '.join(df.columns[:-1])}")
            print(f"   • Target: {df.columns[-1]}")
            
            print(f"\n🔍 Sample data:")
            print(df.head())
            
            input("\n🚀 Press Enter to train the regression model with Libra...")
            
            # Libra one-line regression
            print("\n⚡ Training regression model with Libra...")
            print("   Code: client.regression(data_path)")
            
            if LIBRA_AVAILABLE:
                model = client.regression(data_path)
                print("✅ Model training completed!")
                print(f"📊 Model performance metrics saved to: {self.results_path}")
            else:
                print("🔄 Simulating model training...")
                print("✅ Model would be trained and evaluated automatically!")
            
            print("\n🎯 What Libra did automatically:")
            print("   ✓ Data preprocessing and cleaning")
            print("   ✓ Feature selection and engineering")
            print("   ✓ Model selection (tried multiple algorithms)")
            print("   ✓ Hyperparameter tuning")
            print("   ✓ Cross-validation")
            print("   ✓ Performance evaluation")
            print("   ✓ Generated visualizations")
            
        except Exception as e:
            print(f"❌ Error in regression demo: {e}")
            print("This might be due to missing dependencies or dataset issues.")
    
    def classification_demo(self):
        """Demonstrate classification with iris species prediction"""
        print("\n" + "="*50)
        print("🎯 CLASSIFICATION DEMO - IRIS SPECIES PREDICTION")
        print("="*50)
        
        dataset_file = "iris_classification.csv"
        if not self.check_dataset(dataset_file):
            return
        
        print("🌸 This demo classifies iris flowers into species based on:")
        print("   • Sepal length and width")
        print("   • Petal length and width")
        print("   • Species: Setosa, Versicolor, Virginica")
        
        input("\n📊 Press Enter to load the dataset and start training...")
        
        try:
            # Load and display dataset info
            data_path = str(self.demo_datasets_path / dataset_file)
            df = pd.read_csv(data_path)
            
            print(f"\n📋 Dataset loaded successfully!")
            print(f"   • Rows: {len(df)}")
            print(f"   • Features: {', '.join(df.columns[:-1])}")
            print(f"   • Classes: {', '.join(df.iloc[:, -1].unique())}")
            
            print(f"\n🔍 Sample data:")
            print(df.head())
            
            input("\n🚀 Press Enter to train the classification model with Libra...")
            
            # Libra one-line classification
            print("\n⚡ Training classification model with Libra...")
            print("   Code: client.classification(data_path)")
            
            if LIBRA_AVAILABLE:
                model = client.classification(data_path)
                print("✅ Model training completed!")
                print(f"📊 Model performance metrics saved to: {self.results_path}")
            else:
                print("🔄 Simulating model training...")
                print("✅ Model would be trained and evaluated automatically!")
            
            print("\n🎯 What Libra did automatically:")
            print("   ✓ Data preprocessing and encoding")
            print("   ✓ Train/test split")
            print("   ✓ Multiple classifier comparison")
            print("   ✓ Feature importance analysis")
            print("   ✓ Confusion matrix generation")
            print("   ✓ Classification report")
            print("   ✓ ROC curves and AUC scores")
            
        except Exception as e:
            print(f"❌ Error in classification demo: {e}")
            print("This might be due to missing dependencies or dataset issues.")
    
    def neural_network_demo(self):
        """Demonstrate neural networks with stock price prediction"""
        print("\n" + "="*50)
        print("🧠 NEURAL NETWORK DEMO - STOCK PRICE PREDICTION")
        print("="*50)
        
        dataset_file = "stock_prices.csv"
        if not self.check_dataset(dataset_file):
            return
        
        print("📈 This demo uses neural networks to predict stock prices based on:")
        print("   • Historical price data")
        print("   • Trading volume")
        print("   • Technical indicators")
        print("   • Time series patterns")
        
        input("\n📊 Press Enter to load the dataset and start training...")
        
        try:
            # Load and display dataset info
            data_path = str(self.demo_datasets_path / dataset_file)
            df = pd.read_csv(data_path)
            
            print(f"\n📋 Dataset loaded successfully!")
            print(f"   • Rows: {len(df)}")
            print(f"   • Columns: {len(df.columns)}")
            print(f"   • Date range: {df.iloc[0, 0]} to {df.iloc[-1, 0]}")
            
            print(f"\n🔍 Sample data:")
            print(df.head())
            
            input("\n🚀 Press Enter to train the neural network with Libra...")
            
            # Libra one-line neural network
            print("\n⚡ Training neural network with Libra...")
            print("   Code: client.neural_network(data_path)")
            
            if LIBRA_AVAILABLE:
                model = client.neural_network(data_path)
                print("✅ Neural network training completed!")
                print(f"📊 Model performance metrics saved to: {self.results_path}")
            else:
                print("🔄 Simulating neural network training...")
                print("✅ Neural network would be trained automatically!")
            
            print("\n🎯 What Libra did automatically:")
            print("   ✓ Time series preprocessing")
            print("   ✓ Sequence generation for LSTM")
            print("   ✓ Neural architecture selection")
            print("   ✓ Automatic hyperparameter tuning")
            print("   ✓ Training with early stopping")
            print("   ✓ Model evaluation and validation")
            print("   ✓ Prediction visualization")
            
        except Exception as e:
            print(f"❌ Error in neural network demo: {e}")
            print("This might be due to missing dependencies or dataset issues.")
    
    def compare_models_demo(self):
        """Compare all three model types"""
        print("\n" + "="*50)
        print("📊 MODEL COMPARISON DEMO")
        print("="*50)
        
        print("🔍 This demo shows how Libra handles different ML tasks:")
        print("\n1. 📈 REGRESSION (Continuous prediction)")
        print("   • Predicts numerical values")
        print("   • Example: House prices, temperatures, sales")
        print("   • Metrics: RMSE, MAE, R²")
        
        print("\n2. 🎯 CLASSIFICATION (Category prediction)")
        print("   • Predicts discrete categories")
        print("   • Example: Species, spam detection, diagnosis")
        print("   • Metrics: Accuracy, Precision, Recall, F1")
        
        print("\n3. 🧠 NEURAL NETWORKS (Complex patterns)")
        print("   • Deep learning for complex relationships")
        print("   • Example: Time series, images, NLP")
        print("   • Metrics: Custom loss functions")
        
        print("\n✨ Libra's Advantage:")
        print("   • Same simple API for all tasks")
        print("   • Automatic algorithm selection")
        print("   • Built-in preprocessing")
        print("   • No ML expertise required")
        
        input("\nPress Enter to continue...")
    
    def learn_about_libra(self):
        """Educational section about Libra"""
        print("\n" + "="*50)
        print("🔍 LEARN ABOUT LIBRA")
        print("="*50)
        
        print("🚀 What is Libra?")
        print("   Libra is a Python library that automates the entire")
        print("   machine learning process in just one line of code.")
        
        print("\n🎯 Key Features:")
        print("   ✓ One-line machine learning")
        print("   ✓ Automatic data preprocessing")
        print("   ✓ Model selection and tuning")
        print("   ✓ Built-in visualizations")
        print("   ✓ No ML background required")
        print("   ✓ Supports regression, classification, neural networks")
        
        print("\n📦 Installation:")
        print("   pip install libra")
        
        print("\n💻 Basic Usage:")
        print("   from libra import client")
        print("   client.regression('data.csv')      # For regression")
        print("   client.classification('data.csv')  # For classification")
        print("   client.neural_network('data.csv')  # For neural networks")
        
        print("\n🌟 Perfect for:")
        print("   • Beginners learning ML")
        print("   • Rapid prototyping")
        print("   • Educational purposes")
        print("   • Quick model baselines")
        print("   • Non-technical users")
        
        print("\n📚 Learn More:")
        print("   • GitHub: https://github.com/Palashio/libra")
        print("   • Documentation: https://libradocs.github.io/")
        
        input("\nPress Enter to continue...")
    
    def run(self):
        """Main demo loop"""
        if not self.display_banner():
            return
        
        while True:
            self.display_menu()
            choice = self.get_user_choice()
            
            if choice == 1:
                self.regression_demo()
            elif choice == 2:
                self.classification_demo()
            elif choice == 3:
                self.neural_network_demo()
            elif choice == 4:
                self.compare_models_demo()
            elif choice == 5:
                self.learn_about_libra()
            elif choice == 6:
                print("\n👋 Thank you for trying the Libra demo!")
                print("🚀 Ready to start your ML journey? Install Libra:")
                print("   pip install libra")
                print("\n🌟 Happy machine learning!")
                break


if __name__ == "__main__":
    demo = LibraDemo()
    demo.run()

