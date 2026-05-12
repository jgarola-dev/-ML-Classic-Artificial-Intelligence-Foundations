"""
Model training and evaluation module
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

class DropoutPredictor:
    def __init__(self, model_type='random_forest', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._create_model()
        self.metrics = {}
        self.feature_importance = None
        self.X_test = None
        self.y_test = None

    def _create_model(self):
        models = {
            'logistic_regression': LogisticRegression(random_state=self.random_state, max_iter=1000, class_weight='balanced'),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state, class_weight='balanced'),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=self.random_state),
            'svm': SVC(kernel='rbf', probability=True, random_state=self.random_state, class_weight='balanced')
        }
        return models.get(self.model_type, RandomForestClassifier(random_state=self.random_state))

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def evaluate(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else 0.5
        }
        return self.metrics

    def get_feature_importance(self, feature_names=None, top_n=10):
        """Importancia segura para Árboles, Lineales y SVM"""
        if feature_names is None or self.X_test is None:
            return None
            
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            elif hasattr(self.model, 'coef_') and self.model.coef_ is not None:
                importances = np.abs(self.model.coef_[0])
            else:
                # Fallback seguro para SVM RBF
                perm = permutation_importance(self.model, self.X_test, self.y_test, 
                                            n_repeats=5, random_state=42, n_jobs=-1)
                importances = perm.importances_mean
                
            df_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
            df_imp['importance'] = df_imp['importance'].clip(lower=0)
            return df_imp.sort_values('importance', ascending=False).head(top_n)
        except Exception:
            return pd.DataFrame({'feature': feature_names[:top_n], 'importance': [0]*top_n})

def compare_models(X_train, X_test, y_train, y_test, feature_names):
    results = {}
    for m in ['logistic_regression', 'random_forest', 'gradient_boosting', 'svm']:
        clf = DropoutPredictor(model_type=m)
        clf.train(X_train, y_train)
        results[m] = {
            'model': clf,
            'metrics': clf.evaluate(X_test, y_test),
            'importance': clf.get_feature_importance(feature_names)
        }
    return results
