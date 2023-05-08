# Link Prediction on Node Pairs

## **Table of Contents**
- [Introduction](#introduction)
- [Environment](#environment)
- [Implementation](#implementation)
  - [Feature Extraction](#feature-extraction)
- [Code](#code)
- [How to run my code?](#how-to-run-my-code)
- [Another idea: Using Neural Network Model to make binary classification](#another-idea-using-neural-network-model-to-make-binary-classification)
  - [Features I used to train the model](#features-i-used-to-train-the-model)
  - [A Simple Model I built](#a-simple-model-i-built)
  - [Code of this method](#code-of-this-method)
  - [How to run my code?](#how-to-run-my-code?)


# Introduction
In this project, I tried many methods to predict the links between node pairs, such as logistic regression, neural networks, and random forest. However, these methods did not work well; the score on the public testing set fell between 0.5 and 0.6, which sucks. And using Weakly Connected Components (WCC) can get a higher score of about 0.7.

After much trial and error, eventually, I found XGBClassifier. It implements the gradient boosting algorithm that uses the XGBoost library. It is a popular algorithm for classification problems due to its speed and performance on many datasets. By setting the hyperparameters, the user can adjust the algorithm's behavior and tune the model for the problem at hand.

# Environment

- macOS Ventura 13.3
- Python 3.10
- Conda 23.1.0

<div style="page-break-after: always;"></div>

# Implementation
The provided training data set comprises `node1`, `node2`, and `label`. To create the graph of it, I will have to separate node pairs depending on `label` and choose those pairs with `label` 1. With the graph, we can glimpse the whole dataset with other metrics calculated by the graph info using EDA(Exploratory Data Analysis).

The following are hyperparameters I set for XGBClassifier:
- `learning_rate=0.2`: The step size shrinkage used in update to prevent overfitting.
- `n_estimators=120`: The number of boosting stages to perform.
- `gamma=2.5`: Minimum loss reduction required to make a further partition on a leaf node of the tree.
- `max_depth=25`: Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.

# How to run my code?

It’s recommended that you use a miniconda environment to run my code.

1. Create a conda env.
    
    ```bash
    $ conda create --name <env_name> python=3.10
    ```
    
2. Activate the env you’ve just created.
    
    ```bash
    $ conda activate <env_name>
    ```
    
3. Install all the required packages.
    
    ```bash
    $ pip install -r requirement.txt
    ```
    
4. Modify the config to load your custom CSV file path
5. Run the python file with the relative path to the test dataset. (Default is the public dataset)
    
    ```bash
    $ python link_prediction_xgb.py
    ```

# Code
Please refer to the [notebook](https://github.com/ghnmqdtg/link-prediction-on-node-pairs/blob/main/link_prediction_xgb.ipynb) for more information.