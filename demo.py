#!/usr/bin/env python3
"""
Libra Interactive Demo

This script demonstrates the core functionality of Libra - an automated machine learning library
that enables ML model building with just one line of code through natural language queries.

Features demonstrated:
- Dataset loading and exploration
- Natural language queries for different ML models
- Neural networks, SVM, and regression models
- Model training automation
- Results visualization
- Interactive user experience

Author: Libra Demo
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any

# Try to import libra, provide helpful error message if not available
try:
    from libra import client
    LIBRA_AVAILABLE = True
except ImportError:
    LIBRA_AVAILABLE = False
    print("⚠️  Libra is not installed. Please install it with: pip install libra")


class LibraDemo:
    """Interactive demo class for showcasing Libra's capabilities."""
    
    def __init__(self):
        self.clients = {}
        self.sample_datasets = {}
        self.demo_results = {}
        
    def print_header(self, title: str, char: str = "="):
        """Print a formatted header."""
        print(f"
        print(f"{title:^60}")
        print(f"{char * 60}
        
    def print_step(self, step: str, description: str):
        """Print a formatted step."""
        print(f"🔹 {step}: {description}")
        
    def wait_for_user(self, message: str = "Press Enter to continue..."):
        """Wait for user input to proceed."""
        input(f"
        
    def create_sample_datasets(self):
        """Create sample datasets for demonstration."""
        print("📊 Creating sample datasets for demonstration...")
        
        # Housing dataset (regression)
        np.random.seed(42)
        n_samples = 1000
        
        housing_data = {
            'bedrooms': np.random.randint(1, 6, n_samples),
            'bathrooms': np.random.randint(1, 4, n_samples),
            'square_feet': np.random.randint(800, 3500, n_samples),
            'age': np.random.randint(0, 50, n_samples),
            'location_score': np.random.uniform(1, 10, n_samples),
            'garage': np.random.choice([0, 1], n_samples),
        }
        
        # Create price based on features (with some noise)
        housing_data['price'] = (
            housing_data['bedrooms'] * 15000 +
            housing_data['bathrooms'] * 10000 +
            housing_data['square_feet'] * 100 +
            (50 - housing_data['age']) * 1000 +
            housing_data['location_score'] * 5000 +
            housing_data['garage'] * 8000 +
            np.random.normal(0, 20000, n_samples)
        )
        
        housing_df = pd.DataFrame(housing_data)
        housing_df.to_csv('housing_demo.csv', index=False)
        self.sample_datasets['housing'] = 'housing_demo.csv'
        
        # Customer dataset (classification)
        customer_data = {
            'age': np.random.randint(18, 80, n_samples),
            'income': np.random.randint(20000, 150000, n_samples),
            'spending_score': np.random.randint(1, 100, n_samples),
            'years_customer': np.random.randint(0, 20, n_samples),
            'num_purchases': np.random.randint(0, 50, n_samples),
        }
        
        # Create churn based on features
        churn_prob = (
            (customer_data['age'] > 60) * 0.3 +
            (customer_data['income'] < 40000) * 0.2 +
            (customer_data['spending_score'] < 30) * 0.4 +
            (customer_data['years_customer'] < 2) * 0.3 +
            np.random.uniform(0, 0.2, n_samples)
        )
        customer_data['will_churn'] = (churn_prob > 0.5).astype(int)
        
        customer_df = pd.DataFrame(customer_data)
        customer_df.to_csv('customer_demo.csv', index=False)
        self.sample_datasets['customer'] = 'customer_demo.csv'
        
        print("✅ Sample datasets created successfully!")
        print(f"   - Housing dataset: {len(housing_df)} samples")
        print(f"   - Customer dataset: {len(customer_df)} samples")
        
    def explore_dataset(self, dataset_name: str, file_path: str):
        """Explore and visualize a dataset."""
        print(f"🔍 Exploring {dataset_name} dataset...")
        
        df = pd.read_csv(file_path)
        
        print(f"
        print(f"
        print(df.head())
        
        print(f"
        print(df.info())
        
        print(f"
        print(df.describe())
        
        # Create visualizations
        plt.figure(figsize=(15, 10))
        
        # Correlation heatmap
        plt.subplot(2, 2, 1)
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title(f'{dataset_name.title()} Dataset - Correlation Matrix')
        
        # Distribution of target variable
        target_col = df.columns[-1]  # Assume last column is target
        plt.subplot(2, 2, 2)
        if df[target_col].dtype in ['int64', 'float64']:
            plt.hist(df[target_col], bins=30, alpha=0.7)
            plt.title(f'Distribution of {target_col}')
            plt.xlabel(target_col)
            plt.ylabel('Frequency')
        else:
            df[target_col].value_counts().plot(kind='bar')
            plt.title(f'Distribution of {target_col}')
            plt.xticks(rotation=45)
        
        # Feature distributions
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]
        for i, col in enumerate(numeric_cols):
            plt.subplot(2, 2, 3 + (i % 2))
            plt.hist(df[col], bins=20, alpha=0.7)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            if i == 1:  # Only show first two additional plots
                break
        
        plt.tight_layout()
        plt.savefig(f'{dataset_name}_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return df
        
    def demonstrate_libra_workflow(self, dataset_name: str, file_path: str, queries: list):
        """Demonstrate Libra's workflow with natural language queries."""
        if not LIBRA_AVAILABLE:
            print("❌ Cannot demonstrate Libra workflow - library not installed")
            return
            
        self.print_header(f"Libra Workflow Demo - {dataset_name.title()}")
        
        print("🚀 Creating Libra client...")
        try:
            libra_client = client(file_path)
            self.clients[dataset_name] = libra_client
            print("✅ Libra client created successfully!")
        except Exception as e:
            print(f"❌ Error creating Libra client: {e}")
            return
            
        self.wait_for_user()
        
        # Demonstrate multiple queries
        for i, query_info in enumerate(queries, 1):
            query_type = query_info['type']
            query_text = query_info['query']
            description = query_info['description']
            
            self.print_step(f"Query {i}", f"{description}")
            print(f"   Query: '{query_text}'")
            print(f"   Type: {query_type}")
            
            try:
                print(f"
                
                if query_type == 'neural_network':
                    result = libra_client.neural_network_query(query_text)
                elif query_type == 'svm':
                    result = libra_client.svm_query(query_text)
                elif query_type == 'regression':
                    result = libra_client.regression_query(query_text)
                else:
                    print(f"❌ Unknown query type: {query_type}")
                    continue
                    
                print("✅ Query executed successfully!")
                
                # Get model information
                info = libra_client.info()
                print(f"
                print(f"   - Model ID: {info.get('id', 'N/A')}")
                print(f"   - Target: {info.get('target', 'N/A')}")
                print(f"   - Accuracy: {info.get('accuracy', 'N/A')}")
                print(f"   - Available plots: {list(info.get('plots', {}).keys())}")
                
                # Store results
                self.demo_results[f"{dataset_name}_{query_type}"] = {
                    'query': query_text,
                    'info': info,
                    'client': libra_client
                }
                
            except Exception as e:
                print(f"❌ Error executing query: {e}")
                
            self.wait_for_user()
            
        # Show all models created
        try:
            models = libra_client.model()
            print(f"
            for model_name in models.keys():
                print(f"   - {model_name}")
        except Exception as e:
            print(f"❌ Error retrieving models: {e}")
            
    def run_housing_demo(self):
        """Run the housing price prediction demo."""
        dataset_name = 'housing'
        file_path = self.sample_datasets[dataset_name]
        
        self.print_header("Housing Price Prediction Demo")
        
        # Explore dataset
        df = self.explore_dataset(dataset_name, file_path)
        self.wait_for_user()
        
        # Define queries for housing dataset
        queries = [
            {
                'type': 'neural_network',
                'query': 'predict the house price using all available features',
                'description': 'Neural network for price prediction'
            },
            {
                'type': 'regression',
                'query': 'model the relationship between house features and price',
                'description': 'Linear regression analysis'
            }
        ]
        
        # Demonstrate Libra workflow
        self.demonstrate_libra_workflow(dataset_name, file_path, queries)
        
    def run_customer_demo(self):
        """Run the customer churn prediction demo."""
        dataset_name = 'customer'
        file_path = self.sample_datasets[dataset_name]
        
        self.print_header("Customer Churn Prediction Demo")
        
        # Explore dataset
        df = self.explore_dataset(dataset_name, file_path)
        self.wait_for_user()
        
        # Define queries for customer dataset
        queries = [
            {
                'type': 'neural_network',
                'query': 'predict customer churn based on their behavior',
                'description': 'Neural network for churn classification'
            },
            {
                'type': 'svm',
                'query': 'classify customers who will churn',
                'description': 'Support Vector Machine classification'
            }
        ]
        
        # Demonstrate Libra workflow
        self.demonstrate_libra_workflow(dataset_name, file_path, queries)
        
    def show_summary(self):
        """Show a summary of all demo results."""
        self.print_header("Demo Summary")
        
        if not self.demo_results:
            print("No demo results to display.")
            return
            
        print("🎯 Models Created:")
        for key, result in self.demo_results.items():
            print(f"
            print(f"   Query: {result['query']}")
            info = result['info']
            print(f"   Target: {info.get('target', 'N/A')}")
            print(f"   Accuracy: {info.get('accuracy', 'N/A')}")
            
        print(f"
        print("🎉 Demo completed successfully!")
        
    def cleanup(self):
        """Clean up temporary files."""
        files_to_remove = ['housing_demo.csv', 'customer_demo.csv', 
                          'housing_exploration.png', 'customer_exploration.png']
        
        for file in files_to_remove:
            if os.path.exists(file):
                os.remove(file)
                
    def run_full_demo(self):
        """Run the complete interactive demo."""
        self.print_header("🚀 Welcome to the Libra Interactive Demo! 🚀")
        
        print("This demo will showcase Libra's automated machine learning capabilities:")
        print("• Dataset loading and exploration")
        print("• Natural language queries for ML models")
        print("• Neural networks, SVM, and regression")
        print("• Automated model training and evaluation")
        print("• Results visualization")
        
        if not LIBRA_AVAILABLE:
            print("
            print("The demo will show dataset exploration but cannot run ML models.")
            print("To install Libra: pip install libra")
            
        self.wait_for_user("Press Enter to start the demo...")
        
        try:
            # Create sample datasets
            self.create_sample_datasets()
            self.wait_for_user()
            
            # Run housing demo
            self.run_housing_demo()
            
            # Run customer demo  
            self.run_customer_demo()
            
            # Show summary
            self.show_summary()
            
        except KeyboardInterrupt:
            print("
        except Exception as e:
            print(f"
        finally:
            # Cleanup
            cleanup_choice = input("
            if cleanup_choice == 'y':
                self.cleanup()
                print("🧹 Temporary files cleaned up.")


def main():
    """Main function to run the demo."""
    demo = LibraDemo()
    demo.run_full_demo()


if __name__ == "__main__":
    main()

