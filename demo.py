#!/usr/bin/env python3
"""
Libra Interactive Demo

A comprehensive interactive demonstration of Libra's automated machine learning capabilities.
This demo showcases how Libra can perform complex ML tasks with just one line of code.

Author: Libra Demo
License: MIT
"""

import os
import sys
import time
import warnings
from typing import Optional, Dict, Any

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def check_installation():
    """Check if required packages are installed and provide installation instructions."""
    print("🔍 Checking installation requirements...")
    print("=" * 60)
    
    required_packages = {
        'libra': 'libra',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'sklearn': 'scikit-learn'
    }
    
    missing_packages = []
    
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {package} - Installed")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print("\n🚨 Missing packages detected!")
        print("Please install the missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        print("\nOr install all requirements at once:")
        print("pip install -r requirements.txt")
        print("\nPress Enter to continue anyway (some features may not work)...")
        input()
    else:
        print("\n✅ All required packages are installed!")
        time.sleep(1)

def display_header():
    """Display the demo header with ASCII art."""
    header = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                           🚀 LIBRA INTERACTIVE DEMO 🚀                      ║
║                                                                              ║
║                    Machine Learning in One Line of Code                     ║
║                                                                              ║
║  Libra automates the entire ML pipeline - from data preprocessing to        ║
║  model training and evaluation - all with simple, intuitive commands.       ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(header)

def display_menu():
    """Display the main menu options."""
    menu = """
📋 DEMO MENU - Choose your ML adventure:

1. 🌸 Classification Demo (Iris Dataset)
2. 🏠 Regression Demo (Boston Housing Dataset)  
3. 🧠 Neural Network Demo (Custom Dataset)
4. 📊 Upload Your Own CSV Dataset
5. 🔄 Compare: Libra vs Traditional ML
6. 📚 Learn: What's Happening Behind the Scenes
7. ⚙️  Advanced: Parameter Customization
8. 📈 Visualization Gallery
9. ❓ Help & Documentation
0. 🚪 Exit Demo

Enter your choice (0-9): """
    return input(menu).strip()

def educational_explanation(topic: str):
    """Provide educational explanations about ML concepts."""
    explanations = {
        'classification': """
🎓 CLASSIFICATION EXPLAINED:

Classification is a supervised learning task where we predict discrete categories or classes.
In our Iris example:
- Input: Flower measurements (sepal length, width, petal length, width)
- Output: Species (Setosa, Versicolor, Virginica)

Libra automatically:
✅ Handles data preprocessing (scaling, encoding)
✅ Selects the best algorithm (Random Forest, SVM, etc.)
✅ Performs hyperparameter tuning
✅ Evaluates model performance
✅ Generates visualizations
        """,
        
        'regression': """
🎓 REGRESSION EXPLAINED:

Regression predicts continuous numerical values.
In our Boston Housing example:
- Input: House features (rooms, crime rate, accessibility, etc.)
- Output: House price (continuous value)

Libra automatically:
✅ Handles missing values and outliers
✅ Feature selection and engineering
✅ Model selection (Linear, Random Forest, XGBoost, etc.)
✅ Cross-validation and performance metrics
✅ Residual analysis and predictions
        """,
        
        'neural_networks': """
🎓 NEURAL NETWORKS EXPLAINED:

Neural networks are inspired by the human brain and excel at complex patterns.
They consist of layers of interconnected nodes (neurons).

Libra's neural networks:
✅ Automatic architecture selection
✅ Optimal activation functions
✅ Regularization to prevent overfitting
✅ Learning rate optimization
✅ Early stopping and checkpointing
        """
    }
    
    print(explanations.get(topic, "Topic not found."))
    input("\nPress Enter to continue...")

def classification_demo():
    """Demonstrate classification using the Iris dataset."""
    print("\n" + "="*60)
    print("🌸 CLASSIFICATION DEMO - Iris Flower Species")
    print("="*60)
    
    educational_explanation('classification')
    
    try:
        # Import required libraries
        from libra import client
        from sklearn.datasets import load_iris
        import pandas as pd
        import matplotlib.pyplot as plt
        
        print("\n📊 Loading Iris dataset...")
        
        # Load and prepare data
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = iris.target_names[iris.target]
        
        print(f"Dataset shape: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head())
        
        print("\n🎯 Target distribution:")
        print(df['species'].value_counts())
        
        # Save dataset temporarily
        df.to_csv('temp_iris.csv', index=False)
        
        print("\n🚀 Training model with Libra (ONE LINE!)...")
        print("Code: client('temp_iris.csv').classification('species')")
        
        # Libra magic happens here
        newClient = client('temp_iris.csv')
        newClient.classification('species')
        
        print("\n✅ Model training completed!")
        print("📈 Libra automatically:")
        print("   • Preprocessed the data")
        print("   • Selected the best algorithm")
        print("   • Tuned hyperparameters")
        print("   • Evaluated performance")
        print("   • Generated visualizations")
        
        # Cleanup
        if os.path.exists('temp_iris.csv'):
            os.remove('temp_iris.csv')
            
    except ImportError:
        print("❌ Libra not installed. Please install with: pip install libra")
    except Exception as e:
        print(f"❌ Error during classification demo: {str(e)}")
    
    input("\nPress Enter to return to main menu...")

def regression_demo():
    """Demonstrate regression using the Boston Housing dataset."""
    print("\n" + "="*60)
    print("🏠 REGRESSION DEMO - Boston Housing Prices")
    print("="*60)
    
    educational_explanation('regression')
    
    try:
        from libra import client
        import pandas as pd
        import numpy as np
        from sklearn.datasets import load_boston
        
        print("\n📊 Loading Boston Housing dataset...")
        
        # Note: Boston housing dataset is deprecated, so we'll create a synthetic version
        # or use an alternative approach
        try:
            boston = load_boston()
            df = pd.DataFrame(boston.data, columns=boston.feature_names)
            df['price'] = boston.target
        except ImportError:
            # Create synthetic housing data if Boston dataset is not available
            print("Creating synthetic housing dataset...")
            np.random.seed(42)
            n_samples = 500
            
            df = pd.DataFrame({
                'rooms': np.random.normal(6, 1, n_samples),
                'age': np.random.uniform(0, 100, n_samples),
                'distance': np.random.exponential(3, n_samples),
                'crime_rate': np.random.exponential(1, n_samples),
                'tax_rate': np.random.uniform(200, 800, n_samples),
            })
            
            # Create synthetic price based on features
            df['price'] = (
                df['rooms'] * 5 +
                (100 - df['age']) * 0.1 +
                -df['distance'] * 2 +
                -df['crime_rate'] * 3 +
                -df['tax_rate'] * 0.01 +
                np.random.normal(0, 5, n_samples) + 20
            )
        
        print(f"Dataset shape: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head())
        
        print(f"\n📈 Price statistics:")
        print(df['price'].describe())
        
        # Save dataset temporarily
        df.to_csv('temp_housing.csv', index=False)
        
        print("\n🚀 Training regression model with Libra (ONE LINE!)...")
        print("Code: client('temp_housing.csv').regression('price')")
        
        # Libra magic happens here
        newClient = client('temp_housing.csv')
        newClient.regression('price')
        
        print("\n✅ Regression model training completed!")
        print("📊 Libra automatically:")
        print("   • Handled data preprocessing")
        print("   • Selected optimal regression algorithm")
        print("   • Performed feature selection")
        print("   • Calculated performance metrics (R², RMSE, MAE)")
        print("   • Generated prediction plots")
        
        # Cleanup
        if os.path.exists('temp_housing.csv'):
            os.remove('temp_housing.csv')
            
    except ImportError:
        print("❌ Libra not installed. Please install with: pip install libra")
    except Exception as e:
        print(f"❌ Error during regression demo: {str(e)}")
    
    input("\nPress Enter to return to main menu...")

def neural_network_demo():
    """Demonstrate neural networks with a custom dataset."""
    print("\n" + "="*60)
    print("🧠 NEURAL NETWORK DEMO - Deep Learning Made Simple")
    print("="*60)
    
    educational_explanation('neural_networks')
    
    try:
        from libra import client
        import pandas as pd
        import numpy as np
        
        print("\n🔬 Creating synthetic dataset for neural network demo...")
        
        # Create a more complex synthetic dataset suitable for neural networks
        np.random.seed(42)
        n_samples = 1000
        
        # Generate features with non-linear relationships
        x1 = np.random.uniform(-2, 2, n_samples)
        x2 = np.random.uniform(-2, 2, n_samples)
        x3 = np.random.uniform(-2, 2, n_samples)
        
        # Create non-linear target with complex interactions
        y = (
            np.sin(x1) * np.cos(x2) + 
            x2**2 * x3 + 
            np.exp(x1 * x3) * 0.1 +
            np.random.normal(0, 0.1, n_samples)
        )
        
        # Convert to classification problem
        y_class = ['Class_A' if val > np.median(y) else 'Class_B' for val in y]
        
        df = pd.DataFrame({
            'feature_1': x1,
            'feature_2': x2,
            'feature_3': x3,
            'target': y_class
        })
        
        print(f"Dataset shape: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head())
        
        print("\n🎯 Class distribution:")
        print(df['target'].value_counts())
        
        # Save dataset temporarily
        df.to_csv('temp_neural.csv', index=False)
        
        print("\n🚀 Training neural network with Libra (ONE LINE!)...")
        print("Code: client('temp_neural.csv').neural_network('target')")
        
        # Libra neural network magic
        newClient = client('temp_neural.csv')
        newClient.neural_network('target')
        
        print("\n✅ Neural network training completed!")
        print("🧠 Libra automatically:")
        print("   • Designed optimal network architecture")
        print("   • Selected activation functions")
        print("   • Implemented regularization")
        print("   • Optimized learning rate")
        print("   • Applied early stopping")
        print("   • Generated training curves")
        
        # Cleanup
        if os.path.exists('temp_neural.csv'):
            os.remove('temp_neural.csv')
            
    except ImportError:
        print("❌ Libra not installed. Please install with: pip install libra")
    except Exception as e:
        print(f"❌ Error during neural network demo: {str(e)}")
    
    input("\nPress Enter to return to main menu...")

def upload_csv_demo():
    """Allow users to upload their own CSV file for analysis."""
    print("\n" + "="*60)
    print("📊 CUSTOM CSV UPLOAD DEMO")
    print("="*60)
    
    print("📁 Upload your own CSV file for ML analysis!")
    print("\nInstructions:")
    print("1. Place your CSV file in the same directory as this demo")
    print("2. Ensure your CSV has a header row with column names")
    print("3. Make sure your target column is clearly identifiable")
    
    filename = input("\nEnter your CSV filename (with .csv extension): ").strip()
    
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    if not os.path.exists(filename):
        print(f"❌ File '{filename}' not found in current directory.")
        print("Please make sure the file exists and try again.")
        input("Press Enter to continue...")
        return
    
    try:
        import pandas as pd
        from libra import client
        
        # Load and inspect the dataset
        df = pd.read_csv(filename)
        print(f"\n📊 Dataset loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        print("\nFirst 5 rows:")
        print(df.head())
        
        print("\nData types:")
        print(df.dtypes)
        
        # Ask user to specify target column
        print(f"\nAvailable columns: {list(df.columns)}")
        target_col = input("Enter the name of your target column: ").strip()
        
        if target_col not in df.columns:
            print(f"❌ Column '{target_col}' not found in dataset.")
            input("Press Enter to continue...")
            return
        
        # Determine if it's classification or regression
        if df[target_col].dtype == 'object' or df[target_col].nunique() < 10:
            task_type = 'classification'
            print(f"\n🎯 Detected: Classification task")
            print(f"Target classes: {df[target_col].unique()}")
        else:
            task_type = 'regression'
            print(f"\n📈 Detected: Regression task")
            print(f"Target range: {df[target_col].min()} to {df[target_col].max()}")
        
        # Ask for confirmation
        proceed = input(f"\nProceed with {task_type}? (y/n): ").strip().lower()
        if proceed != 'y':
            return
        
        print(f"\n🚀 Training {task_type} model with Libra...")
        print(f"Code: client('{filename}').{task_type}('{target_col}')")
        
        # Run Libra
        newClient = client(filename)
        if task_type == 'classification':
            newClient.classification(target_col)
        else:
            newClient.regression(target_col)
        
        print(f"\n✅ {task_type.title()} model training completed!")
        print("🎉 Your custom dataset has been analyzed with Libra!")
        
    except ImportError:
        print("❌ Required libraries not installed.")
    except Exception as e:
        print(f"❌ Error processing your CSV: {str(e)}")
    
    input("\nPress Enter to return to main menu...")

def compare_traditional_vs_libra():
    """Compare Libra approach with traditional ML workflow."""
    print("\n" + "="*60)
    print("🔄 LIBRA vs TRADITIONAL ML COMPARISON")
    print("="*60)
    
    comparison = """
📊 TRADITIONAL ML WORKFLOW (50+ lines of code):

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data.csv')

# Handle missing values
df = df.dropna()

# Encode categorical variables
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    if col != 'target':
        df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Train best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

# Visualizations
plt.figure(figsize=(10, 6))
# ... more plotting code ...
```

⚡ LIBRA APPROACH (1 line of code):

```python
from libra import client

client('data.csv').classification('target')
```

🎯 WHAT LIBRA DOES AUTOMATICALLY:
✅ Data preprocessing and cleaning
✅ Feature engineering and selection
✅ Algorithm selection and comparison
✅ Hyperparameter optimization
✅ Cross-validation and evaluation
✅ Visualization generation
✅ Model interpretation
✅ Performance reporting

💡 TIME SAVED: Hours → Seconds
💡 CODE REDUCED: 50+ lines → 1 line
💡 EXPERTISE REQUIRED: Expert → Beginner
💡 ERROR PRONE: High → Minimal
    """
    
    print(comparison)
    input("\nPress Enter to return to main menu...")

def behind_the_scenes():
    """Explain what happens behind the scenes in Libra."""
    print("\n" + "="*60)
    print("📚 WHAT'S HAPPENING BEHIND THE SCENES")
    print("="*60)
    
    explanation = """
🔍 LIBRA'S AUTOMATED ML PIPELINE:

1. 📊 DATA ANALYSIS & PREPROCESSING:
   • Automatic data type detection
   • Missing value imputation strategies
   • Outlier detection and handling
   • Feature scaling and normalization
   • Categorical encoding (one-hot, label, target)

2. 🔧 FEATURE ENGINEERING:
   • Automatic feature selection
   • Polynomial feature generation
   • Interaction term creation
   • Dimensionality reduction (PCA, LDA)
   • Feature importance ranking

3. 🤖 MODEL SELECTION:
   • Algorithm comparison and benchmarking
   • Ensemble method evaluation
   • Deep learning architecture search
   • Cross-validation strategies
   • Performance metric optimization

4. ⚙️ HYPERPARAMETER OPTIMIZATION:
   • Grid search and random search
   • Bayesian optimization
   • Genetic algorithms
   • Early stopping mechanisms
   • Learning curve analysis

5. 📈 MODEL EVALUATION:
   • Multiple performance metrics
   • Confusion matrices and ROC curves
   • Residual analysis for regression
   • Feature importance plots
   • Model interpretation (SHAP values)

6. 🎨 VISUALIZATION GENERATION:
   • Data distribution plots
   • Correlation heatmaps
   • Model performance charts
   • Prediction vs actual plots
   • Feature importance visualizations

7. 📋 AUTOMATED REPORTING:
   • Model performance summary
   • Best algorithm recommendation
   • Feature importance rankings
   • Prediction confidence intervals
   • Model deployment readiness

🧠 INTELLIGENCE LEVELS:

👶 BEGINNER MODE:
   • Fully automated decisions
   • Best practices applied
   • Explanatory outputs

👨‍💻 INTERMEDIATE MODE:
   • Parameter customization
   • Algorithm preferences
   • Evaluation metric selection

🔬 EXPERT MODE:
   • Full pipeline control
   • Custom preprocessing
   • Advanced configurations
    """
    
    print(explanation)
    input("\nPress Enter to return to main menu...")

def parameter_customization():
    """Show advanced parameter customization options."""
    print("\n" + "="*60)
    print("⚙️ ADVANCED PARAMETER CUSTOMIZATION")
    print("="*60)
    
    customization_info = """
🎛️ LIBRA CUSTOMIZATION OPTIONS:

📊 DATA PREPROCESSING:
   • preprocessing_options: Custom preprocessing steps
   • imputation_strategy: 'mean', 'median', 'mode', 'drop'
   • scaling_method: 'standard', 'minmax', 'robust', 'none'
   • encoding_strategy: 'onehot', 'label', 'target', 'auto'

🤖 MODEL SELECTION:
   • algorithms: Specify which algorithms to try
   • ensemble_methods: Enable/disable ensemble techniques
   • cross_validation: Custom CV strategies
   • scoring_metric: 'accuracy', 'f1', 'roc_auc', 'mse', 'r2'

⚙️ HYPERPARAMETER TUNING:
   • optimization_method: 'grid', 'random', 'bayesian'
   • n_trials: Number of optimization trials
   • timeout: Maximum optimization time
   • early_stopping: Enable early stopping

📈 OUTPUT CONTROL:
   • verbose: Control output verbosity
   • save_model: Automatically save trained models
   • generate_plots: Enable/disable visualizations
