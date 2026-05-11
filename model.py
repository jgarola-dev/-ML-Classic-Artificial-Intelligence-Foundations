"""
Model training and evaluation module
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

class DropoutPredictor:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        
    def train(self, X_train, y_train):
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        elif self.model_type == 'gradient_boost':
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        elif self.model_type == 'svm':
            self.model = SVC(probability=True, random_state=42, class_weight='balanced')
        
        self.model.fit(X_train, y_train)
        return self
    
    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None
        }
        return metrics
    
    def get_feature_importance(self, X_test, y_test, feature_names, top_n=10):
        """Importancia segura para árboles, lineales y SVM"""
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                importances = np.abs(self.model.coef_[0])
            else:
                # Fallback: Permutation Importance (funciona para cualquier modelo)
                perm_imp = permutation_importance(self.model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
                importances = perm_imp.importances_mean
                
            df_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
            return df_imp.sort_values('importance', ascending=False).head(top_n)
        except Exception:
            return None
