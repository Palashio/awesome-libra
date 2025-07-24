#!/usr/bin/env python3
"""
Libra Interactive Demo

This demo showcases the core functionality of Libra, an ergonomic machine learning
library that enables users to build ML models with natural language queries.

Features demonstrated:
- Client object creation with sample datasets
- Natural language ML queries (neural_network_query, svm_query)
- Result visualization and model information access
- Interactive exploration of different ML algorithms

Run this demo interactively or execute sections to explore Libra's capabilities.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up plotting style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")

def print_section_header(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_step(step_num, description):
    """Print a formatted step"""
    print(f"\n📍 Step {step_num}: {description}")
    print("-" * 50)

def create_sample_datasets():
    """Create sample datasets for demonstration"""
    print_section_header("CREATING SAMPLE DATASETS")
    
    # Create demo directory if it doesn't exist
    demo_dir = Path("demo")
    demo_dir.mkdir(exist_ok=True)
    
    # Dataset 1: Housing prices (regression)
    print_step(1, "Creating housing prices dataset")
    np.random.seed(42)
    n_samples = 1000
    
    housing_data = {
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'square_feet': np.random.randint(800, 3500, n_samples),
        'age': np.random.randint(0, 50, n_samples),
        'location_score': np.random.uniform(1, 10, n_samples),
        'median_households': np.random.randint(100, 2000, n_samples)
    }
    
    # Create realistic price based on features
    housing_data['price'] = (
        housing_data['bedrooms'] * 15000 +
        housing_data['bathrooms'] * 8000 +
        housing_data['square_feet'] * 120 +
        (50 - housing_data['age']) * 1000 +
        housing_data['location_score'] * 5000 +
        np.random.normal(0, 20000, n_samples)
    )
    
    housing_df = pd.DataFrame(housing_data)
    housing_path = demo_dir / "housing_data.csv"
    housing_df.to_csv(housing_path, index=False)
    print(f"✅ Created housing dataset: {housing_path}")
    print(f"   Shape: {housing_df.shape}")
    print(f"   Columns: {list(housing_df.columns)}")
    
    # Dataset 2: Ocean proximity classification
    print_step(2, "Creating ocean proximity dataset")
    
    proximity_data = {
        'latitude': np.random.uniform(32, 42, n_samples),
        'longitude': np.random.uniform(-125, -114, n_samples),
        'housing_median_age': np.random.randint(1, 52, n_samples),
        'total_rooms': np.random.randint(500, 8000, n_samples),
        'population': np.random.randint(100, 5000, n_samples),
        'median_income': np.random.uniform(0.5, 15, n_samples)
    }
    
    # Create proximity categories based on longitude (simplified)
    proximity_categories = []
    for lng in proximity_data['longitude']:
        if lng > -117:
            proximity_categories.append('INLAND')
        elif lng > -121:
            proximity_categories.append('NEAR BAY')
        else:
            proximity_categories.append('<1H OCEAN')
    
    proximity_data['ocean_proximity'] = proximity_categories
    
    proximity_df = pd.DataFrame(proximity_data)
    proximity_path = demo_dir / "ocean_proximity_data.csv"
    proximity_df.to_csv(proximity_path, index=False)
    print(f"✅ Created ocean proximity dataset: {proximity_path}")
    print(f"   Shape: {proximity_df.shape}")
    print(f"   Columns: {list(proximity_df.columns)}")
    
    return housing_path, proximity_path

def demo_libra_basics():
    """Demonstrate basic Libra functionality"""
    print_section_header("LIBRA BASICS DEMONSTRATION")
    
    try:
        from libra import client
        print("✅ Libra imported successfully!")
    except ImportError:
        print("❌ Libra not found. Please install it with: pip install libra")
        print("   For this demo, we'll show you what the code would look like.")
        return None
    
    print_step(1, "Understanding the Libra Client")
    print("""
The core of Libra is the 'client' object. Here's how it works:

1. Create a client with your dataset
2. Ask natural language questions about your data
3. Get ML models, visualizations, and insights automatically

Let's see it in action!
    """)
    
    return client

def demo_neural_network_query(client_class, housing_path):
    """Demonstrate neural network queries"""
    print_section_header("NEURAL NETWORK QUERY DEMO")
    
    if client_class is None:
        print("Simulating Libra neural network query...")
        print("""
# What the code would look like:
from libra import client

# Create client with housing dataset
housing_client = client('demo/housing_data.csv')

# Ask a natural language question
housing_client.neural_network_query('please model the median number of households')

# Get results
results = housing_client.info()
print(f"Model accuracy: {results['accuracy']}")
print(f"Available plots: {list(results['plots'].keys())}")
        """)
        return
    
    print_step(1, "Creating client with housing dataset")
    try:
        housing_client = client_class(str(housing_path))
        print(f"✅ Client created with dataset: {housing_path}")
    except Exception as e:
        print(f"❌ Error creating client: {e}")
        return
    
    print_step(2, "Running neural network query")
    try:
        print("Query: 'please model the median number of households'")
        housing_client.neural_network_query('please model the median number of households')
        print("✅ Neural network model trained successfully!")
        
        # Get model information
        info = housing_client.info()
        print(f"\n📊 Model Results:")
        print(f"   - Model type: {info.get('model', 'N/A')}")
        print(f"   - Target variable: {info.get('target', 'N/A')}")
        print(f"   - Accuracy: {info.get('accuracy', 'N/A')}")
        print(f"   - Available plots: {list(info.get('plots', {}).keys())}")
        
    except Exception as e:
        print(f"❌ Error running neural network query: {e}")

def demo_svm_query(client_class, proximity_path):
    """Demonstrate SVM queries"""
    print_section_header("SVM QUERY DEMO")
    
    if client_class is None:
        print("Simulating Libra SVM query...")
        print("""
# What the code would look like:
from libra import client

# Create client with ocean proximity dataset
ocean_client = client('demo/ocean_proximity_data.csv')

# Ask about classification
ocean_client.svm_query('predict the proximity to the ocean')

# Get results
results = ocean_client.info()
print(f"Classification accuracy: {results['accuracy']}")
        """)
        return
    
    print_step(1, "Creating client with ocean proximity dataset")
    try:
        ocean_client = client_class(str(proximity_path))
        print(f"✅ Client created with dataset: {proximity_path}")
    except Exception as e:
        print(f"❌ Error creating client: {e}")
        return
    
    print_step(2, "Running SVM query")
    try:
        print("Query: 'predict the proximity to the ocean'")
        ocean_client.svm_query('predict the proximity to the ocean')
        print("✅ SVM model trained successfully!")
        
        # Get model information
        info = ocean_client.info()
        print(f"\n📊 Model Results:")
        print(f"   - Model type: {info.get('model', 'N/A')}")
        print(f"   - Target variable: {info.get('target', 'N/A')}")
        print(f"   - Accuracy: {info.get('accuracy', 'N/A')}")
        print(f"   - Number of classes: {info.get('num_classes', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error running SVM query: {e}")

def main():
    """Main demo function"""
    print_section_header("🚀 WELCOME TO THE LIBRA INTERACTIVE DEMO 🚀")
