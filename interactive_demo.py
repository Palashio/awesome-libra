#!/usr/bin/env python3
"""
Interactive Libra Demo - Machine Learning in One Line of Code

This demo showcases Libra's core functionality for automated machine learning.
Libra enables end-to-end ML with automatic preprocessing and model selection.

Features demonstrated:
- One-line classification with built-in datasets
- Automatic data preprocessing
- Automatic model selection and optimization
- Easy-to-use API for users of all experience levels
"""

import sys
import os

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import libra
        import pandas as pd
        import numpy as np
        from sklearn.datasets import load_iris, load_wine
        print("✅ All dependencies are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nTo install required packages, run:")
        print("pip install libra pandas numpy scikit-learn")
        return False

def demo_iris_classification():
    """Demonstrate Libra classification with the Iris dataset."""
    print("\n" + "="*60)
    print("🌸 IRIS CLASSIFICATION DEMO")
    print("="*60)
    
    try:
        from libra import client
        from sklearn.datasets import load_iris
        import pandas as pd
        
        # Load the iris dataset
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        
        print("📊 Dataset Info:")
        print(f"   - Features: {len(iris.feature_names)}")
        print(f"   - Samples: {len(df)}")
        print(f"   - Classes: {len(iris.target_names)} ({', '.join(iris.target_names)})")
        
        # Save dataset to CSV for Libra
        df.to_csv('iris_data.csv', index=False)
        print("   - Dataset saved as 'iris_data.csv'")
        
        print("\n🚀 Running Libra Classification (One Line of Code!):")
        print("   newClient = client('iris_data.csv')")
        print("   newClient.classification()")
        
        # This is the "one line" that does everything!
        newClient = client('iris_data.csv')
        newClient.classification()
        
        print("\n✅ Classification completed!")
        print("   Libra automatically:")
        print("   - Preprocessed the data")
        print("   - Selected the best model")
        print("   - Trained and evaluated the model")
        print("   - Generated performance metrics")
        
    except Exception as e:
        print(f"❌ Error during classification: {e}")
        print("This might be due to Libra version compatibility or missing dependencies.")

def demo_wine_classification():
    """Demonstrate Libra classification with the Wine dataset."""
    print("\n" + "="*60)
    print("🍷 WINE CLASSIFICATION DEMO")
    print("="*60)
    
    try:
        from libra import client
        from sklearn.datasets import load_wine
        import pandas as pd
        
        # Load the wine dataset
        wine = load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df['target'] = wine.target
        
        print("📊 Dataset Info:")
        print(f"   - Features: {len(wine.feature_names)}")
        print(f"   - Samples: {len(df)}")
        print(f"   - Classes: {len(wine.target_names)} wine types")
        
        # Save dataset to CSV for Libra
        df.to_csv('wine_data.csv', index=False)
        print("   - Dataset saved as 'wine_data.csv'")
        
        print("\n🚀 Running Libra Classification (One Line of Code!):")
        print("   newClient = client('wine_data.csv')")
        print("   newClient.classification()")
        
        # This is the "one line" that does everything!
        newClient = client('wine_data.csv')
        newClient.classification()
        
        print("\n✅ Classification completed!")
        
    except Exception as e:
        print(f"❌ Error during classification: {e}")

def main():
    """Main function to run the interactive demo."""
    print("🤖 LIBRA INTERACTIVE DEMO")
    print("Machine Learning in One Line of Code!")
    print("-" * 50)
    
    if not check_dependencies():
        return
    
    print("\nThis demo showcases Libra's automated machine learning capabilities.")
    print("Libra is designed for users of all experience levels - from beginners")
    print("to ML engineers - enabling collaboration on the same tasks.\n")
    
    # Run classification demos
    demo_iris_classification()
    demo_wine_classification()
    
    print("\n" + "="*60)
    print("🎉 DEMO COMPLETED!")
    print("="*60)
    print("Key takeaways:")
    print("✨ Machine learning in just one line of code")
    print("✨ Automatic data preprocessing")
    print("✨ Automatic model selection and optimization")
    print("✨ No prior ML experience required")
    print("✨ Works for users of all technical backgrounds")
    
    print("\nGenerated files:")
    if os.path.exists('iris_data.csv'):
        print("📄 iris_data.csv - Iris dataset")
    if os.path.exists('wine_data.csv'):
        print("📄 wine_data.csv - Wine dataset")
    
    print("\nFor more advanced examples, check out 'libra_interactive_demo.ipynb'!")

if __name__ == "__main__":
    main()

