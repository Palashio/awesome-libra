#!/usr/bin/env python3
"""
Interactive Libra Machine Learning Demo

This demo showcases the automated machine learning capabilities of the Libra library.
Libra allows users to build machine learning models with minimal code - often just one line!

Features demonstrated:
- Neural network queries
- SVM queries  
- Decision tree queries
- K-means clustering
- Sample dataset generation
- Interactive user experience
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Optional

def check_and_install_libra():
    """Check if libra is installed, and provide installation instructions if not."""
    try:
        import libra
        print("✅ Libra is already installed!")
        return True
    except ImportError:
        print("❌ Libra is not installed.")
        print("
        print("pip install libra")
        print("
        print("pip install -r requirements.txt")
        
        install_now = input("
        if install_now == 'y':
            try:
                import subprocess
                print("Installing Libra...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "libra"])
                print("✅ Libra installed successfully!")
                return True
            except subprocess.CalledProcessError:
                print("❌ Failed to install Libra. Please install manually.")
                return False
        return False

def create_sample_datasets():
    """Create sample datasets for demonstration purposes."""
    print("
    
    # Dataset 1: Housing prices (regression)
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic housing data
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.randint(1, 4, n_samples)
    sqft = np.random.normal(2000, 500, n_samples)
    age = np.random.randint(0, 50, n_samples)
    
    # Create price based on features with some noise
    price = (bedrooms * 50000 + bathrooms * 30000 + sqft * 100 - age * 1000 + 
             np.random.normal(0, 20000, n_samples))
    price = np.maximum(price, 100000)  # Minimum price
    
    housing_data = pd.DataFrame({
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft': sqft,
        'age': age,
        'price': price
    })
    
    housing_data.to_csv('housing_data.csv', index=False)
    print("✅ Created housing_data.csv (regression dataset)")
    
    # Dataset 2: Customer classification
    n_customers = 800
    age = np.random.randint(18, 80, n_customers)
    income = np.random.normal(50000, 20000, n_customers)
    spending_score = np.random.randint(1, 100, n_customers)
    
    # Create customer segments based on features
    segments = []
    for i in range(n_customers):
        if income[i] > 60000 and spending_score[i] > 70:
            segments.append('Premium')
        elif income[i] > 40000 and spending_score[i] > 50:
            segments.append('Standard')
        else:
            segments.append('Budget')
    
    customer_data = pd.DataFrame({
        'age': age,
        'income': income,
        'spending_score': spending_score,
        'segment': segments
    })
    
    customer_data.to_csv('customer_data.csv', index=False)
    print("✅ Created customer_data.csv (classification dataset)")
    
    # Dataset 3: Simple clustering data
    n_points = 500
    cluster_data = pd.DataFrame({
        'x': np.random.normal(0, 2, n_points),
        'y': np.random.normal(0, 2, n_points),
        'feature1': np.random.normal(10, 3, n_points),
        'feature2': np.random.normal(5, 2, n_points)
    })
    
    cluster_data.to_csv('cluster_data.csv', index=False)
    print("✅ Created cluster_data.csv (clustering dataset)")
    
    return ['housing_data.csv', 'customer_data.csv', 'cluster_data.csv']

def display_menu():
    """Display the main menu options."""
    print("
    print("🚀 LIBRA INTERACTIVE MACHINE LEARNING DEMO")
    print("="*60)
    print("Choose a demo to run:")
    print("1. 🏠 Neural Network Regression (Housing Prices)")
    print("2. 👥 SVM Classification (Customer Segments)")
    print("3. 🌳 Decision Tree Classification (Customer Segments)")
    print("4. 🎯 K-Means Clustering (Data Grouping)")
    print("5. 📊 Show Dataset Information")
    print("6. 🔄 Regenerate Sample Datasets")
    print("0. ❌ Exit")
    print("="*60)

def run_neural_network_demo(libra_client):
    """Demonstrate neural network regression on housing data."""
    print("
    print("-" * 40)
    print("Using housing_data.csv to predict house prices...")
    print("Features: bedrooms, bathrooms, sqft, age")
    print("Target: price")
    
    try:
        client = libra_client('housing_data.csv')
        print("
        client.neural_network_query('predict the price of houses')
        
        print("✅ Neural network training completed!")
        print("
        info = client.info()
        print(f"- Model Type: {info.get('model', 'N/A')}")
        print(f"- Target: {info.get('target', 'N/A')}")
        print(f"- Accuracy: {info.get('accuracy', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ Error running neural network demo: {str(e)}")
        return False

def run_svm_demo(libra_client):
    """Demonstrate SVM classification on customer data."""
    print("
    print("-" * 40)
    print("Using customer_data.csv to classify customer segments...")
    print("Features: age, income, spending_score")
    print("Target: segment (Premium/Standard/Budget)")
    
    try:
        client = libra_client('customer_data.csv')
        print("
        client.svm_query('classify customer segments')
        
        print("✅ SVM training completed!")
        print("
        info = client.info()
        print(f"- Model Type: {info.get('model', 'N/A')}")
        print(f"- Target: {info.get('target', 'N/A')}")
        print(f"- Number of Classes: {info.get('num_classes', 'N/A')}")
        print(f"- Accuracy: {info.get('accuracy', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ Error running SVM demo: {str(e)}")
        return False

def run_decision_tree_demo(libra_client):
    """Demonstrate decision tree classification on customer data."""
    print("
    print("-" * 40)
    print("Using customer_data.csv to classify customer segments...")
    print("Features: age, income, spending_score")
    print("Target: segment (Premium/Standard/Budget)")
    
    try:
        client = libra_client('customer_data.csv')
        print("
        client.decision_tree_query('predict customer segment')
        
        print("✅ Decision tree training completed!")
        print("
        info = client.info()
        print(f"- Model Type: {info.get('model', 'N/A')}")
        print(f"- Target: {info.get('target', 'N/A')}")
        print(f"- Number of Classes: {info.get('num_classes', 'N/A')}")
        print(f"- Accuracy: {info.get('accuracy', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ Error running decision tree demo: {str(e)}")
        return False

def run_clustering_demo(libra_client):
    """Demonstrate K-means clustering on sample data."""
    print("
    print("-" * 40)
    print("Using cluster_data.csv to find data groups...")
    print("Features: x, y, feature1, feature2")
    
    try:
        client = libra_client('cluster_data.csv')
        print("
        client.kmeans_clustering_query('find clusters in the data')
        
        print("✅ K-means clustering completed!")
        print("
        info = client.info()
        print(f"- Model Type: {info.get('model', 'N/A')}")
        print(f"- Number of Clusters: {info.get('num_classes', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ Error running clustering demo: {str(e)}")
        return False

def show_dataset_info():
    """Display information about the sample datasets."""
    print("
    print("-" * 40)
    
    datasets = [
        ('housing_data.csv', 'Regression'),
        ('customer_data.csv', 'Classification'),
        ('cluster_data.csv', 'Clustering')
    ]
    
    for filename, task_type in datasets:
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            print(f"
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Sample data:")
            print(f"   {df.head(2).to_string(index=False)}")
        else:
            print(f"

def main():
    """Main demo function."""
    print("Welcome to the Libra Interactive Machine Learning Demo!")
    print("This demo showcases automated ML with minimal code.")
    
    # Check if Libra is installed
    if not check_and_install_libra():
        print("
        return
    
    # Import libra after confirming installation
    from libra import client
    
    # Create sample datasets
    datasets = create_sample_datasets()
    
    while True:
        display_menu()
        choice = input("
        
        if choice == '0':
            print("
            print("Visit https://github.com/Palashio/libra for more information.")
            break
        elif choice == '1':
            run_neural_network_demo(client)
        elif choice == '2':
            run_svm_demo(client)
        elif choice == '3':
            run_decision_tree_demo(client)
        elif choice == '4':
            run_clustering_demo(client)
        elif choice == '5':
            show_dataset_info()
        elif choice == '6':
            create_sample_datasets()
        else:
            print("❌ Invalid choice. Please enter a number between 0-6.")
        
        if choice in ['1', '2', '3', '4']:
            input("

if __name__ == "__main__":
    main()

