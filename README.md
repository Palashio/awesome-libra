# awesome-libra
A curated list of all things related to libra.

These are only original pieces of media around libra. Re-writes, reshares, and blog features are not included.

## Installation & Prerequisites

### Installation

Install the latest release version:

```bash
pip install -U libra
```

### Prerequisites

- Python 3.6 or higher
- pandas for data manipulation
- scikit-learn for machine learning algorithms
- TensorFlow/Keras for neural networks (automatically installed with libra)

## Interactive Python Examples

Libra is an ergonomic machine learning library that automates the entire ML process with just a few lines of code. Below are comprehensive examples showing how to use Libra's core functionality.

### Basic Client Setup

The core functionality of Libra works through the `client` object. Create a new client object for every dataset you want to analyze:

```python
from libra import client

# Create a client with your dataset
newClient = client('path/to/your/dataset.csv')
```

### Neural Network Queries

Libra automatically determines whether to use regression or classification based on your target column type.

#### Regression Example

```python
# Predict a continuous numerical value
newClient.neural_network_query('predict median house value')

# The model is automatically stored in the client
print("Available models:", list(newClient.models.keys()))
# Output: ['regression_ANN']
```

#### Classification Example

```python
# Predict a categorical value
newClient.neural_network_query('predict ocean proximity')

# Now you have both models
print("Available models:", list(newClient.models.keys()))
# Output: ['regression_ANN', 'classification_ANN']
```

### Accessing Model Information

Get comprehensive information about all your models:

```python
# Get detailed information about all models and results
model_info = newClient.info()

# View the structure of information available
print("Information keys:", list(model_info.keys()))
# Output: ['id', 'model', 'num_classes', 'plots', 'target', 'preprocessor', 
#          'interpreter', 'test_data', 'losses', 'accuracy']
```

### Model Performance Analysis

Use the `analyze()` function to evaluate your models and generate performance plots:

```python
# Analyze regression model performance
newClient.analyze(model='regression_ANN')

# Check what metrics were added
regression_model = newClient.models['regression_ANN']
print("Regression metrics:", [key for key in regression_model.keys() if key in ['MSE', 'MAE']])
# Output: ['MSE', 'MAE']

# Analyze classification model performance
newClient.analyze(model='classification_ANN')

# Check classification metrics and plots
classification_model = newClient.models['classification_ANN']
print("Classification plots:", list(classification_model['plots'].keys()))
# Output: ['roc_curve', 'confusion_matrix']

print("Classification scores:", list(classification_model['scores'].keys()))
# Output: ['recall_score', 'precision_score', 'f1_score']
```

### Complete Workflow Walkthrough

Here's a comprehensive walkthrough showing the complete Libra workflow with multiple query types:

#### Step 1: Dataset Loading

Start by loading your dataset and creating a client object:

```python
from libra import client

# Load the California housing dataset
newClient = client('housing.csv')
print("Dataset loaded successfully!")
```

#### Step 2: Multiple Query Types

Libra supports various query types for different machine learning approaches:

```python
# Neural Network Query (automatically chooses regression/classification)
newClient.neural_network_query('predict median house value')  # Regression
newClient.neural_network_query('predict ocean proximity')     # Classification

# Specific Neural Network Queries
newClient.regression_query_ann('predict median house value')
newClient.classification_query_ann('predict ocean proximity')

# Support Vector Machine Query
newClient.svm_query('predict ocean proximity')

print("All models trained successfully!")
```

#### Step 3: Accessing the Models Dictionary

All trained models are stored in the client's models dictionary:

```python
# View all available models
print("Available models:", list(newClient.models.keys()))
# Output: ['regression_ANN', 'classification_ANN', 'svm']

# Access specific model information
regression_model = newClient.models['regression_ANN']
classification_model = newClient.models['classification_ANN']
svm_model = newClient.models['svm']

print("Regression model keys:", list(regression_model.keys()))
print("Classification model keys:", list(classification_model.keys()))
print("SVM model keys:", list(svm_model.keys()))
```

#### Step 4: Model Evaluation with analyze()

Use the analyze() function to evaluate each model's performance:

```python
# Analyze regression model
newClient.analyze(model='regression_ANN')
print("Regression analysis complete - MSE and MAE metrics added")

# Analyze classification model
newClient.analyze(model='classification_ANN')
print("Classification analysis complete - ROC curve, confusion matrix, and scores added")

# Analyze SVM model
newClient.analyze(model='svm')
print("SVM analysis complete - performance metrics and plots generated")
```

#### Step 5: Retrieving Comprehensive Results

Get all information about your models and results:

```python
# Get comprehensive information about all models
comprehensive_results = newClient.info()

print("Comprehensive results structure:")
print("Available information:", list(comprehensive_results.keys()))
print(f"Total models trained: {len(newClient.models)}")

# Access specific model performance metrics
for model_name in newClient.models.keys():
    print(f"
```

## Video Tutorials
[Machine Learning in One Line of Code](https://www.youtube.com/watch?v=N_T_ljj5vc4) by Ahmad Bazzi.

[Introduction to Machine Learning using Libra](https://www.youtube.com/watch?v=kQrIrm1XKc0&t=4s) by Palash Shah.

[Libra - Your Data Talks Meetup](https://www.youtube.com/watch?v=2149kjp97KI&t=407s) by Palash Shah

## Articles
[Libra: A Python tool that Automates Machine Learning Process in a Few Lines of Code](https://www.marktechpost.com/2020/07/28/libra-a-python-tool-that-automates-machine-learning-process-in-a-few-lines-of-code/) by marktechpost. 

[One liner Machine learning and Deep Learning using Libra](https://towardsdatascience.com/machine-learning-and-deep-learning-in-one-liner-using-libra-7eef4023618f) by Ali Aryan.

[Create a complex Machine Learning model in one line with Libra](https://towardsdatascience.com/create-a-complex-machine-learning-model-in-one-line-with-libra-a253e05d15a1) by Cornellius Yudha Wijaya.

[Fully Automated Machine Learning in One-Liners](https://medium.com/@gagan.2492/fully-automated-machine-learning-in-one-liners-5925ba994b48) by Gagandeep Singh.

[Machine Learning in One-Minute with Libra](https://medium.com/@pranavnt5/machine-learning-in-one-minute-with-libra-783dcd393f7f) by Pranav Teegavarapu.

[The Ultimate Out-of-the-box Automated Python Model Selection Methods](https://towardsdatascience.com/the-ultimate-out-of-the-box-automated-python-model-selection-methods-f2188472d2a) by Philippe Bouaziz, PhD.

[AutoML: the Good, the Bad, and the Ugly](https://www.compris.xyz/post/automl) by Vagif Aliyev

[Create a neural network in one-line of code](https://www.machinelearningmadeeasy.com/libra) by Siddhant Chadha.

[Libra : Fully Automated Machine Learning in One-Liners](https://medium.com/@ravi07/libra-fully-automated-machine-learning-in-one-liners-27ca352339ed) by Ravi.

[Scikit-learn, TensorFlow, PyTorch, Keras… but what about Libra?](https://towardsdatascience.com/scikit-learn-tensorflow-pytorch-keras-but-what-about-libra-a5102c2d834d) by Ugo Loobuyck. 

## Webinars

[Become a machine learning expert](https://www.meetup.com/Cloud-Computing-AI-Big-Data-and-Machine-Learning/events/272040486/) at Cloud Computing, AI, Big Data. (Almost filled up over 250+ signups).

[Become a machine learning expert in 45 minutes](https://zmurl.com/hyphora-libra) at Hyphora

## Other

[#1 Trending Project on Made with ML in August](https://madewithml.com/projects/2122/libra/)





