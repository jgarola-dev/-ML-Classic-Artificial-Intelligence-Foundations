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
    classification_report
)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

class DropoutPredictor:
    """Modelo ML para predicción de abandono escolar"""
    def __init__(self, model_type='random_forest', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._create_model()
        self.metrics = {}
        self.feature_importance = None
        self.feature_names = None
        self.X_test = None
        self.y_test = None

    def _create_model(self):
        """Crea el modelo según el tipo especificado"""
        models = {
            'logistic_regression': LogisticRegression(random_state=self.random_state, max_iter=1000, class_weight='balanced'),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state, class_weight='balanced'),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=self.random_state),
            'svm': SVC(kernel='rbf', probability=True, random_state=self.random_state, class_weight='balanced')
        }
        return models.get(self.model_type, RandomForestClassifier(random_state=self.random_state))

    def train(self, X_train, y_train):
        """Entrena el modelo"""
        self.model.fit(X_train, y_train)
        self.feature_names = getattr(X_train, 'columns', None)
        
        # Obtener importancia de características de forma segura
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            self.feature_importance = np.abs(self.model.coef_[0])

    def predict(self, X):
        """Realiza predicciones"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predicciones con probabilidades"""
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """Evalúa el modelo con múltiples métricas"""
        self.X_test = X_test
        self.y_test = y_test
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
        """Retorna las características más importantes"""
        if self.feature_importance is not None and feature_names is not None:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.feature_importance
            }).sort_values('importance', ascending=False)
            return importance_df.head(top_n)
            
        # Fallback para SVM u otros modelos sin importancia explícita
        if self.X_test is not None and self.y_test is not None and feature_names is not None:
            try:
                perm = permutation_importance(self.model, self.X_test, self.y_test, 
                                            n_repeats=5, random_state=42, n_jobs=-1)
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': perm.importances_mean
                }).sort_values('importance', ascending=False)
                return importance_df.head(top_n)
            except Exception:
                return pd.DataFrame({'feature': feature_names[:top_n], 'importance': [0]*top_n})
        return None

    def save_model(self, filepath):
        """Guarda el modelo entrenado"""
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        """Carga un modelo entrenado"""
        self.model = joblib.load(filepath)
        
    def get_metrics(self):
        """Retorna las métricas de evaluación"""
        return self.metrics

    def get_recommendation(self, pred, prob=None):
        """Proporciona recomendaciones basadas en predicción"""
        if prob is None:
            return "🚨 RIESGO DE ABANDONO DETECTADO" if pred == 1 else "✅ BAJO RIESGO DE ABANDONO"
            
        if prob > 0.7:
            return "🚨 RIESGO ALTO de abandono escolar"
        elif prob > 0.4:
            return "⚠️ RIESGO MEDIO de abandono escolar"
        else:
            return "✅ BAJO RIESGO de abandono escolar"

def compare_models(X_train, X_test, y_train, y_test, feature_names=None):
    """Entrena y compara múltiples modelos"""
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
            'metrics': metrics,
            'importance': predictor.get_feature_importance(feature_names, top_n=10)
        }
        
    return results

if __name__ == "__main__":
    print("✅ Model module loaded successfully")
