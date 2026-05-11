"""
Model training and evaluation module
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

class DropoutPredictor:
    def __init__(self, model_type='random_forest', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._create_model()
        self.metrics = {}
        self.feature_importance = None
        self.feature_names = None

    def _create_model(self):
        models = {
            'logistic_regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=self.random_state),
            'svm': SVC(kernel='rbf', probability=True, random_state=self.random_state)
        }
        return models.get(self.model_type, RandomForestClassifier(random_state=self.random_state))

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.feature_names = getattr(X_train, 'columns', None)
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        return self.metrics

    def get_feature_importance(self, feature_names=None, top_n=10):
        if self.feature_importance is None:
            return None
        names = feature_names if feature_names is not None else [f'Feature_{i}' for i in range(len(self.feature_importance))]
        importance_df = pd.DataFrame({'feature': names, 'importance': self.feature_importance})
        return importance_df.sort_values('importance', ascending=False).head(top_n)

def compare_models(X_train, X_test, y_train, y_test):
    model_types = ['logistic_regression', 'random_forest', 'gradient_boosting', 'svm']
    results = {}
    for mt in model_types:
        predictor = DropoutPredictor(model_type=mt)
        predictor.train(X_train, y_train)
        results[mt] = {'model': predictor, 'metrics': predictor.evaluate(X_test, y_test)}
    return results
