"""
Model training and evaluation module for student dropout prediction
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

class DropoutPredictor:
    """ML Model for student dropout prediction"""
    def __init__(self, model_type='random_forest', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._create_model()
        self.metrics = {}
        self.feature_importance = None
        self.feature_names = None

    def _create_model(self):
        """Creates the model based on the specified type"""
        models = {
            'logistic_regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=self.random_state),
            'svm': SVC(kernel='rbf', probability=True, random_state=self.random_state)
        }
        return models.get(self.model_type, RandomForestClassifier(random_state=self.random_state))

    def train(self, X_train, y_train):
        """Trains the model"""
        self.model.fit(X_train, y_train)
        self.feature_names = getattr(X_train, 'columns', None)
        
        # Get feature importance safely
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            self.feature_importance = np.abs(self.model.coef_[0])

    def predict(self, X):
        """Makes predictions"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Makes predictions with probabilities"""
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """Evaluates the model with multiple metrics"""
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
        """Returns the most important features"""
        if self.feature_importance is None:
            return None
            
        if feature_names is not None:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.feature_importance
            }).sort_values('importance', ascending=False)
        else:
            importance_df = pd.DataFrame({
                'feature': [f'Feature_{i}' for i in range(len(self.feature_importance))],
                'importance': self.feature_importance
            }).sort_values('importance', ascending=False)
            
        return importance_df.head(top_n)

    def save_model(self, filepath):
        """Saves the trained model"""
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        """Loads a trained model"""
        self.model = joblib.load(filepath)

    def get_metrics(self):
        """Returns evaluation metrics"""
        return self.metrics

    def get_recommendation(self, pred, prob=None):
        """Provides recommendations based on prediction"""
        if prob is None:
            if pred == 1:
                return "🚨 RIESGO DE ABANDONO DETECTADO"
            else:
                return "✅ BAJO RIESGO DE ABANDONO"
                
        if prob > 0.7:
            return "🚨 RIESGO ALTO de abandono escolar"
        elif prob > 0.4:
            return "⚠️ RIESGO MEDIO de abandono escolar"
        else:
            return "✅ BAJO RIESGO de abandono escolar"


def compare_models(X_train, X_test, y_train, y_test):
    """Trains and compares multiple models"""
    model_types = [
        'logistic_regression',
        'random_forest',
        'gradient_boosting',
        'svm'
    ]
    
    results = {}
    
    for model_type in model_types:
        predictor = DropoutPredictor(model_type=model_type)
        predictor.train(X_train, y_train)
        metrics = predictor.evaluate(X_test, y_test)
        results[model_type] = {
            'model': predictor,
            'metrics': metrics
        }
        
    return results

if __name__ == "__main__":
    print("✅ Model module loaded successfully")
