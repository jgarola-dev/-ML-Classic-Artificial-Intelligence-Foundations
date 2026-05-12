"""
Model training, evaluation & tuning module
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

class DropoutPredictor:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = self._create_model()
        self.metrics = {}
        self.feature_importance = None

    def _create_model(self):
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
        }
        return models.get(self.model_type, RandomForestClassifier(random_state=42))

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        # Extraer importancia de forma segura
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            self.feature_importance = np.abs(self.model.coef_[0])
            
    def evaluate(self, X_test, y_test):
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

    def get_feature_importance_df(self, feature_names, top_n=10):
        if self.feature_importance is None or feature_names is None:
            return None
        df = pd.DataFrame({'feature': feature_names, 'importance': self.feature_importance})
        return df.sort_values('importance', ascending=False).head(top_n)

def compare_and_tune(X_train, y_train, X_test, y_test, feature_names):
    """Entrena 4 modelos + GridSearchCV en Random Forest"""
    models_to_train = ['logistic_regression', 'random_forest', 'gradient_boosting', 'svm']
    results = {}
    
    # 1. Entrenar modelos base
    for m in models_to_train:
        clf = DropoutPredictor(model_type=m)
        clf.train(X_train, y_train)
        results[m] = {
            'model': clf,
            'metrics': clf.evaluate(X_test, y_test),
            'importance': clf.get_feature_importance_df(feature_names, top_n=10)
        }
        
    # 2. Hyperparameter Tuning (Requisito URV)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced']
    }
    
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42), param_grid,
        scoring='f1', cv=3, n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)
    
    tuning_results = {
        'best_params': grid.best_params_,
        'best_f1': grid.best_score_,
        'baseline_f1': results['random_forest']['metrics']['f1'],
        'improvement': ((grid.best_score_ - results['random_forest']['metrics']['f1']) / results['random_forest']['metrics']['f1']) * 100
    }
    
    # Actualizar modelo RF con mejores params
    results['random_forest']['model'] = DropoutPredictor('random_forest')
    results['random_forest']['model'].model = grid.best_estimator_
    results['random_forest']['model'].train(X_train, y_train)
    results['random_forest']['metrics'] = results['random_forest']['model'].evaluate(X_test, y_test)
    
    return results, tuning_results
