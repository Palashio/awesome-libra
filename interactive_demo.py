#!/usr/bin/env python3
"""
Interactive Demo for Libra Machine Learning Library

This demo showcases the key features of Libra, an automated machine learning library
that allows users to build ML models using natural language queries.

Libra Repository: https://github.com/Palashio/libra
Documentation: http://libradocs.org/
"""

import os
import sys

try:
    from libra import client
except ImportError:
    print("❌ Libra library not found!")
    print("Please install it using: pip install libra")
    print("Or install all dependencies: pip install -r requirements.txt")
    sys.exit(1)


class LibraDemo:
    """Interactive demonstration of Libra machine learning capabilities."""
    
    def __init__(self, data_path="sample_data.csv"):
        """Initialize the demo with a dataset."""
        self.data_path = data_path
        self.client = None
        self.models_trained = []
        
    def initialize_client(self):
        """Initialize the Libra client with the sample dataset."""
        if not os.path.exists(self.data_path):
            print(f"❌ Dataset file '{self.data_path}' not found!")
            print("Please ensure sample_data.csv exists in the current directory.")
            return False
            
        try:
            print(f"🔄 Loading dataset: {self.data_path}")
            self.client = client(self.data_path)
            print("✅ Dataset loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def display_menu(self):
        """Display the main menu options."""
        print("\n" + "="*60)
        print("🚀 LIBRA MACHINE LEARNING DEMO")
        print("="*60)
        print("Choose a machine learning algorithm to demonstrate:")
        print()
        print("1. 🧠 Neural Network (Auto-detects regression/classification)")
        print("2. 🌳 Decision Tree")
        print("3. 🎯 Support Vector Machine (SVM)")
        print("4. 📊 K-Means Clustering")
        print("5. 🚀 XGBoost")
        print("6. 🔍 K-Nearest Neighbors")
        print("7. 📈 View Model Results")
        print("8. 🎛️  Interactive Dashboard")
        print("9. ❓ Help & Information")
        print("0. 🚪 Exit")
        print("="*60)
    
    def neural_network_demo(self):
        """Demonstrate neural network functionality."""
        print("\n🧠 NEURAL NETWORK DEMO")
        print("-" * 40)
        print("Libra's neural_network_query() automatically detects whether to use")
        print("regression or classification based on your target variable.")
        print()
        
        # Regression example
        print("📊 Running regression example...")
        try:
            self.client.neural_network_query('predict price', epochs=5)
            print("✅ Regression neural network trained successfully!")
            self.models_trained.append('regression_ANN')
        except Exception as e:
            print(f"❌ Error training regression model: {e}")
        
        # Classification example
        print("\n📊 Running classification example...")
        try:
            self.client.neural_network_query('predict location', epochs=5)
            print("✅ Classification neural network trained successfully!")
            self.models_trained.append('classification_ANN')
        except Exception as e:
            print(f"❌ Error training classification model: {e}")
    
    def decision_tree_demo(self):
        """Demonstrate decision tree functionality."""
        print("\n🌳 DECISION TREE DEMO")
        print("-" * 40)
        print("Decision trees are interpretable models that make decisions")
        print("by splitting data based on feature values.")
        print()
        
        try:
            self.client.decision_tree_query('predict location')
            print("✅ Decision tree trained successfully!")
            self.models_trained.append('decision_tree')
        except Exception as e:
            print(f"❌ Error training decision tree: {e}")
    
    def svm_demo(self):
        """Demonstrate SVM functionality."""
        print("\n🎯 SUPPORT VECTOR MACHINE DEMO")
        print("-" * 40)
        print("SVMs find optimal boundaries between different classes")
        print("by maximizing the margin between data points.")
        print()
        
        try:
            self.client.svm_query('predict location')
            print("✅ SVM trained successfully!")
            self.models_trained.append('svm')
        except Exception as e:
            print(f"❌ Error training SVM: {e}")
    
    def kmeans_demo(self):
        """Demonstrate K-means clustering functionality."""
        print("\n📊 K-MEANS CLUSTERING DEMO")
        print("-" * 40)
        print("K-means clustering groups similar data points together")
        print("without using target labels (unsupervised learning).")
        print()
        
        try:
            self.client.kmeans_clustering_query(clusters=3)
            print("✅ K-means clustering completed successfully!")
            self.models_trained.append('k_means_clustering')
        except Exception as e:
            print(f"❌ Error running K-means clustering: {e}")
    
    def xgboost_demo(self):
        """Demonstrate XGBoost functionality."""
        print("\n🚀 XGBOOST DEMO")
        print("-" * 40)
        print("XGBoost is a powerful gradient boosting algorithm")
        print("that often achieves state-of-the-art results.")
        print()
        
        try:
            self.client.xgboost_query('predict location')
            print("✅ XGBoost trained successfully!")
            self.models_trained.append('xgboost')
        except Exception as e:
            print(f"❌ Error training XGBoost: {e}")
    
    def knn_demo(self):
        """Demonstrate K-Nearest Neighbors functionality."""
        print("\n🔍 K-NEAREST NEIGHBORS DEMO")
        print("-" * 40)
        print("KNN classifies data points based on the majority class")
        print("of their k nearest neighbors in the feature space.")
        print()
        
        try:
            self.client.nearest_neighbor_query('predict location')
            print("✅ K-Nearest Neighbors trained successfully!")
            self.models_trained.append('nearest_neighbor')
        except Exception as e:
            print(f"❌ Error training KNN: {e}")
    
    def view_results(self):
        """Display results from trained models."""
        print("\n📈 MODEL RESULTS")
        print("-" * 40)
        
        if not self.models_trained:
            print("No models have been trained yet. Please run some algorithms first.")
            return
        
        print(f"Available models: {', '.join(self.models_trained)}")
        print("\nModel dictionary keys:", list(self.client.models.keys()))
        
        # Show basic information about each model
        for model_name in self.models_trained:
            if model_name in self.client.models:
                print(f"\n📊 {model_name.upper()}:")
                model_info = self.client.models[model_name]
                print(f"  - Keys available: {list(model_info.keys())}")
    
    def show_dashboard(self):
        """Launch the interactive dashboard."""
        print("\n🎛️  INTERACTIVE DASHBOARD")
        print("-" * 40)
        print("Launching Libra's interactive dashboard...")
        
        try:
            self.client.dashboard()
            print("✅ Dashboard launched successfully!")
        except Exception as e:
            print(f"❌ Error launching dashboard: {e}")
    
    def show_help(self):
