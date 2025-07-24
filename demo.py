#!/usr/bin/env python3
"""
Interactive Libra Demo - Automated Machine Learning Made Simple

This demo showcases Libra's key features including one-line machine learning
automation, data preprocessing, model selection, and analysis.

Libra is designed for users of all technical levels - from beginners to ML engineers.
No prior experience in data preprocessing or complex ML frameworks is required!

Author: Libra Community
Website: https://libradocs.org/
GitHub: https://github.com/Palashio/libra
"""

import sys
import os
import time
from typing import Optional

# Color codes for better terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_colored(text: str, color: str = Colors.ENDC) -> None:
    """Print colored text to terminal."""
    print(f"{color}{text}{Colors.ENDC}")

def print_header(text: str) -> None:
    """Print a formatted header."""
    print_colored("\n" + "="*60, Colors.HEADER)
    print_colored(f"  {text}", Colors.HEADER + Colors.BOLD)
    print_colored("="*60, Colors.HEADER)

def print_step(step: int, text: str) -> None:
    """Print a formatted step."""
    print_colored(f"\n📋 Step {step}: {text}", Colors.OKBLUE + Colors.BOLD)

def print_success(text: str) -> None:
    """Print success message."""
    print_colored(f"✅ {text}", Colors.OKGREEN)

def print_warning(text: str) -> None:
    """Print warning message."""
    print_colored(f"⚠️  {text}", Colors.WARNING)

def print_error(text: str) -> None:
    """Print error message."""
    print_colored(f"❌ {text}", Colors.FAIL)

def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    print_step(1, "Checking Dependencies")
    
    required_packages = {
        'libra': 'libra',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn'
    }
    
    missing_packages = []
    
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print_success(f"{package} is installed")
        except ImportError:
            missing_packages.append(pip_name)
            print_error(f"{package} is not installed")
    
    if missing_packages:
        print_warning("\nMissing dependencies detected!")
        print_colored("Please install the missing packages using:", Colors.WARNING)
        print_colored(f"pip install {' '.join(missing_packages)}", Colors.OKCYAN)
        print_colored("\nOr install all dependencies at once:", Colors.WARNING)
        print_colored("pip install -r requirements.txt", Colors.OKCYAN)
        return False
    
    print_success("All dependencies are installed!")
    return True

def wait_for_user(prompt: str = "Press Enter to continue...") -> None:
    """Wait for user input."""
    print_colored(f"\n{prompt}", Colors.OKCYAN)
    input()

def get_user_choice(prompt: str, choices: list) -> str:
    """Get user choice from a list of options."""
    while True:
        print_colored(f"\n{prompt}", Colors.OKCYAN)
        for i, choice in enumerate(choices, 1):
            print_colored(f"  {i}. {choice}", Colors.OKBLUE)
        
        try:
            choice_num = int(input("\nEnter your choice (number): "))
            if 1 <= choice_num <= len(choices):
                return choices[choice_num - 1]
            else:
                print_error("Invalid choice. Please try again.")
        except ValueError:
            print_error("Please enter a valid number.")

def demo_classification():
    """Demonstrate Libra's classification capabilities."""
    print_header("CLASSIFICATION DEMO - Iris Flower Prediction")
    
    print_colored("""
🌸 Classification Demo: Iris Flower Species Prediction

In this demo, we'll use the famous Iris dataset to predict flower species
based on measurements like petal length, sepal width, etc.

What makes Libra special?
• Automatic data preprocessing
• Intelligent model selection
• Built-in performance analysis
• All in just ONE line of code!
    """, Colors.OKGREEN)
    
    wait_for_user()
    
    try:
        from libra import client
        from sklearn.datasets import load_iris
        import pandas as pd
        
        print_step(1, "Loading the Iris dataset")
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = iris.target_names[iris.target]
        
        # Save to CSV for Libra
        df.to_csv('iris_demo.csv', index=False)
        print_success("Dataset loaded and saved as 'iris_demo.csv'")
        print_colored(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns", Colors.OKBLUE)
        
        print_step(2, "The Magic Happens Here - ONE LINE OF CODE!")
        print_colored("""
Here's the Libra magic - everything automated in one line:

    client('iris_demo.csv', drop=['species'], target='species')

This single line will:
✨ Load and analyze your data
✨ Handle missing values and preprocessing
✨ Try multiple ML algorithms automatically
✨ Select the best performing model
✨ Generate performance reports and visualizations
        """, Colors.OKCYAN)
        
        wait_for_user("Ready to see the magic? Press Enter to run the model...")
        
        print_colored("🚀 Running Libra's automated ML pipeline...", Colors.HEADER)
        
        # Run Libra classification
        model = client('iris_demo.csv', drop=['species'], target='species')
        
        print_success("Classification model trained successfully!")
        print_colored("Libra automatically selected the best algorithm and hyperparameters!", Colors.OKGREEN)
        
        # Clean up
        if os.path.exists('iris_demo.csv'):
            os.remove('iris_demo.csv')
            
    except Exception as e:
        print_error(f"Demo failed: {str(e)}")
        print_colored("This might be due to missing dependencies or data issues.", Colors.WARNING)

def demo_regression():
    """Demonstrate Libra's regression capabilities."""
    print_header("REGRESSION DEMO - House Price Prediction")
    
    print_colored("""
🏠 Regression Demo: House Price Prediction

In this demo, we'll predict house prices based on features like
size, location, number of bedrooms, etc.

Regression is used when you want to predict continuous numerical values
(like prices, temperatures, or sales figures).
    """, Colors.OKGREEN)
    
    wait_for_user()
    
    try:
        from libra import client
        from sklearn.datasets import fetch_california_housing
        import pandas as pd
        
        print_step(1, "Loading the California Housing dataset")
        housing = fetch_california_housing()
        df = pd.DataFrame(housing.data, columns=housing.feature_names)
        df['price'] = housing.target
        
        # Take a smaller sample for demo purposes
        df_sample = df.sample(n=1000, random_state=42)
        df_sample.to_csv('housing_demo.csv', index=False)
        
        print_success("Dataset loaded and saved as 'housing_demo.csv'")
        print_colored(f"Dataset shape: {df_sample.shape[0]} rows, {df_sample.shape[1]} columns", Colors.OKBLUE)
        
        print_step(2, "One-Line Regression with Libra")
        print_colored("""
Again, just one line of code for complete regression analysis:

    client('housing_demo.csv', drop=['price'], target='price')

Libra will automatically:
🔍 Analyze feature relationships
📊 Handle data scaling and normalization  
🤖 Test multiple regression algorithms
📈 Provide accuracy metrics and predictions
        """, Colors.OKCYAN)
        
        wait_for_user("Ready to predict house prices? Press Enter...")
        
        print_colored("🚀 Running regression analysis...", Colors.HEADER)
        
        # Run Libra regression
        model = client('housing_demo.csv', drop=['price'], target='price')
        
        print_success("Regression model trained successfully!")
        print_colored("Your model can now predict house prices based on the input features!", Colors.OKGREEN)
        
        # Clean up
        if os.path.exists('housing_demo.csv'):
            os.remove('housing_demo.csv')
            
    except Exception as e:
        print_error(f"Demo failed: {str(e)}")
        print_colored("This might be due to missing dependencies or data issues.", Colors.WARNING)

def demo_neural_networks():
    """Demonstrate Libra's neural network capabilities."""
    print_header("NEURAL NETWORKS DEMO - Deep Learning Made Simple")
    
    print_colored("""
🧠 Neural Networks Demo: Deep Learning Without the Complexity

Neural networks are powerful for complex pattern recognition tasks.
Traditionally, setting up neural networks requires extensive knowledge
of architectures, layers, activation functions, etc.

With Libra, you get enterprise-grade neural networks in one line!
    """, Colors.OKGREEN)
    
    wait_for_user()
    
    try:
        from libra import client
        from sklearn.datasets import make_classification
        import pandas as pd
        
        print_step(1, "Creating a synthetic dataset for neural network demo")
        
        # Create a more complex dataset suitable for neural networks
        X, y = make_classification(
            n_samples=2000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=3,
            random_state=42
        )
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(20)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        df.to_csv('neural_demo.csv', index=False)
        print_success("Complex dataset created and saved as 'neural_demo.csv'")
        print_colored(f"Dataset: {df.shape[0]} samples, {df.shape[1]-1} features, 3 classes", Colors.OKBLUE)
        
        print_step(2, "Neural Network Training - Still Just One Line!")
        print_colored("""
Even for neural networks, Libra keeps it simple:

    client('neural_demo.csv', drop=['target'], target='target')

Behind the scenes, Libra will:
🧠 Design an optimal neural network architecture
⚡ Configure activation functions and layers
🎯 Implement proper regularization techniques
📊 Monitor training progress and prevent overfitting
        """, Colors.OKCYAN)
        
        wait_for_user("Ready to train a neural network? Press Enter...")
        
        print_colored("🚀 Training neural network (this may take a moment)...", Colors.HEADER)
        
        # Run Libra neural network
        model = client('neural_demo.csv', drop=['target'], target='target')
        
        print_success("Neural network trained successfully!")
        print_colored("Libra automatically optimized the network architecture for your data!", Colors.OKGREEN)
        
        # Clean up
        if os.path.exists('neural_demo.csv'):
            os.remove('neural_demo.csv')
            
    except Exception as e:
        print_error(f"Demo failed: {str(e)}")
        print_colored("This might be due to missing dependencies or computational requirements.", Colors.WARNING)

def show_libra_benefits():
    """Show the key benefits of using Libra."""
    print_header("WHY CHOOSE LIBRA?")
    
    print_colored("""
🚀 LIBRA'S KEY ADVANTAGES:

1. 📚 BEGINNER-FRIENDLY
   • No need to learn complex ML frameworks
   • Clear documentation and examples
   • Works for users of all skill levels

2. ⚡ INCREDIBLY FAST
   • One line of code for complete ML pipelines
   • Automatic preprocessing and feature engineering
   • No manual hyperparameter tuning needed

3. 🎯 INTELLIGENT AUTOMATION
   • Automatically selects the best algorithms
   • Handles missing data and outliers
   • Generates comprehensive performance reports

4. 🔧 PRODUCTION-READY
   • Enterprise-grade model performance
   • Built-in model validation and testing
   • Easy integration with existing workflows

5. 🌍 COLLABORATIVE
   • Technical and non-technical users can work together
   • Standardized approach across teams
   • Reproducible results
    """, Colors.OKGREEN)

def main():
    """Main demo function."""
    print_colored("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║           🚀 WELCOME TO THE LIBRA INTERACTIVE DEMO 🚀        ║
║                                                              ║
║              Automated Machine Learning Made Simple          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """, Colors.HEADER + Colors.BOLD)
    
    print_colored("""
Hello! Welcome to the interactive Libra demonstration.

Libra is a Python library that makes machine learning accessible to everyone.
Whether you're a complete beginner or an experienced data scientist,
Libra can help you build powerful ML models with minimal code.

This demo will show you three core ML capabilities:
• Classification (predicting categories)
• Regression (predicting numbers)  
• Neural Networks (deep learning)
    """, Colors.OKBLUE)
    
    # Check dependencies first
    if not check_dependencies():
        print_colored("\nPlease install the required dependencies and run the demo again.", Colors.WARNING)
        return
    
    wait_for_user()
    
    # Main demo loop
    while True:
        demos = [
            "Classification Demo (Iris Flowers)",
            "Regression Demo (House Prices)", 
            "Neural Networks Demo (Deep Learning)",
            "Show Libra Benefits",
            "Exit Demo"
