import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

"""
    Define various models for hypertuning and experiments
"""
# Logistic regression, Decision Tree    
def logisticRegressionModel():
    return LogisticRegression(random_state=42)

def decisionTreeModel():
    return tree.DecisionTreeClassifier()

# Random Forest, SVM
def randomForestModel():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42
    )

def svm():
    return

# Gradient Boosing, Neural Network, Bayesian Network
def xgboostModel():
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        enable_categorical=True,
        eval_metric="logloss"
    )
    
def neuralNetworkModel():
    return

def bayesianNetworkModel():
    return