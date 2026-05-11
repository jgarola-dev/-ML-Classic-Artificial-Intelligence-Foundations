"""
Feature Engineering Pipeline for Student Dropout Analysis
Comprehensive analysis of features, statistics, and visualizations
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from scipy.stats import skew, kurtosis, pointbiserialr, spearmanr
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineeringPipeline:
    """
    Comprehensive Feature Engineering Pipeline for exploratory data analysis
    """
    
    def __init__(self, df):
        """Initialize pipeline with dataframe"""
        self.df = df.copy()
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.target_col = 'Status' if 'Status' in df.columns else None
    
    def get_descriptive_statistics(self):
        """Generate descriptive statistics for numeric columns"""
        stats = self.df[self.numeric_cols].describe().T
        stats['skewness'] = self.df[self.numeric_cols].skew()
        stats['kurtosis'] = self.df[self.numeric_cols].kurtosis()
        stats['variance'] = self.df[self.numeric_cols].var()
        stats['q25'] = self.df[self.numeric_cols].quantile(0.25)
        stats['q75'] = self.df[self.numeric_cols].quantile(0.75)
        stats['iqr'] = stats['q75'] - stats['q25']
        return stats
    
    def get_categorical_statistics(self):
        """Analyze categorical features - frequencies and class imbalance"""
        categorical_stats = {}
        
        for col in self.categorical_cols:
            value_counts = self.df[col].value_counts()
            percentages = (value_counts / len(self.df) * 100).round(2)
            
            categorical_stats[col] = {
                'counts': value_counts,
                'percentages': percentages,
                'unique_values': len(value_counts),
                'missing': self.df[col].isna().sum(),
                'missing_pct': (self.df[col].isna().sum() / len(self.df) * 100).round(2),
                'dominant_class': value_counts.index[0] if len(value_counts) > 0 else None,
                'dominant_class_pct': percentages.iloc[0] if len(percentages) > 0 else None
            }
        
        return categorical_stats
    
    def plot_categorical_distributions(self):
        """Create bar and pie charts for categorical variables"""
        figs = {}
        
        for col in self.categorical_cols[:10]:  # Limit to first 10
            value_counts = self.df[col].value_counts()
            
            # Bar chart
            fig_bar = px.bar(
                x=value_counts.index.astype(str),
                y=value_counts.values,
                title=f"Distribución: {col} (Frecuencias Absolutas)",
                labels={'x': col, 'y': 'Frecuencia'},
                color=value_counts.values,
                color_continuous_scale='Viridis'
            )
            
            # Pie chart
            fig_pie = px.pie(
                values=value_counts.values,
                names=value_counts.index.astype(str),
                title=f"Distribución: {col} (Porcentajes)"
            )
            
            figs[f"{col}_bar"] = fig_bar
            figs[f"{col}_pie"] = fig_pie
        
        return figs
    
    def plot_numeric_distributions(self):
        """Create histograms and box plots for numeric variables"""
        figs = {}
        
        for col in self.numeric_cols[:15]:  # Limit to first 15
            # Histogram
            fig_hist = px.histogram(
                self.df,
                x=col,
                title=f"Distribución: {col}",
                nbins=50,
                color_discrete_sequence=['#636EFA']
            )
            
            # Box plot
            fig_box = px.box(
                self.df,
                y=col,
                title=f"Box Plot: {col}",
                color_discrete_sequence=['#636EFA']
            )
            
            figs[f"{col}_hist"] = fig_hist
            figs[f"{col}_box"] = fig_box
        
        return figs
    
    def plot_pairwise_relationships(self, var1, var2):
        """Plot relationship between two numeric variables"""
        fig = px.scatter(
            self.df,
            x=var1,
            y=var2,
            title=f"Relación: {var1} vs {var2}",
            trendline="ols",
            trendline_color_override="red",
            labels={'x': var1, 'y': var2}
        )
        
        fig.update_traces(marker=dict(size=5, opacity=0.6))
        return fig
    
    def plot_scatter_matrix(self, sample_cols=None):
        """Create scatter matrix for all numeric variables"""
        if sample_cols is None:
            sample_cols = self.numeric_cols[:8]  # Limit to 8 columns for performance
        else:
            sample_cols = [col for col in sample_cols if col in self.numeric_cols]
        
        if len(sample_cols) < 2:
            return None
        
        fig = px.scatter_matrix(
            self.df[sample_cols],
            dimensions=sample_cols,
            title="Matriz de Dispersión - Todas las Relaciones Pareadas",
            labels={col: col for col in sample_cols},
            color_discrete_sequence=['#636EFA']
        )
        
        fig.update_traces(marker=dict(size=3, opacity=0.5))
        return fig
    
    def get_correlation_matrix(self):
        """Calculate correlation matrix"""
        return self.df[self.numeric_cols].corr()
    
    def plot_correlation_heatmap(self, figsize=(12, 10)):
        """Create heatmap of correlation matrix"""
        corr_matrix = self.get_correlation_matrix()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text:.2f}',
            textfont={"size": 8},
            colorbar=dict(title="Correlación")
        ))
        
        fig.update_layout(
            title="Matriz de Correlación - Heatmap",
            xaxis_tickangle=-45,
            height=600,
            width=800
        )
        
        return fig
    
    def get_top_correlations(self, top_n=15):
        """Get top N strongest correlations (excluding diagonal)"""
        corr_matrix = self.get_correlation_matrix()
        
        # Get upper triangle of correlation matrix
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Variable_1': corr_matrix.columns[i],
                    'Variable_2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        df_corr = pd.DataFrame(corr_pairs)
        df_corr['Abs_Correlation'] = df_corr['Correlation'].abs()
        df_corr = df_corr.sort_values('Abs_Correlation', ascending=False)
        
        return df_corr.head(top_n)
    
    def analyze_class_imbalance(self):
        """Analyze target variable class distribution"""
        if self.target_col is None:
            return None
        
        class_counts = self.df[self.target_col].value_counts()
        class_percentages = (class_counts / len(self.df) * 100).round(2)
        
        if len(class_counts) < 2:
            return None
        
        analysis = {
            'class_counts': class_counts,
            'class_percentages': class_percentages,
            'imbalance_ratio': class_counts.max() / class_counts.min() if class_counts.min() > 0 else float('inf'),
            'is_imbalanced': (class_counts.max() / class_counts.min()) > 1.5 if class_counts.min() > 0 else True,
            'minority_class': class_counts.idxmin(),
            'majority_class': class_counts.idxmax(),
            'minority_count': class_counts.min(),
            'majority_count': class_counts.max()
        }
        
        return analysis
    
    def plot_class_imbalance(self):
        """Visualize class distribution"""
        if self.target_col is None:
            return None, None
        
        class_counts = self.df[self.target_col].value_counts()
        
        # Bar chart
        fig_bar = px.bar(
            x=class_counts.index.astype(str),
            y=class_counts.values,
            title=f"Distribución de Clases - {self.target_col}",
            labels={'x': self.target_col, 'y': 'Frecuencia'},
            color=class_counts.values,
            color_continuous_scale='Viridis'
        )
        
        # Pie chart
        fig_pie = px.pie(
            values=class_counts.values,
            names=class_counts.index.astype(str),
            title=f"Proporción de Clases - {self.target_col}"
        )
        
        return fig_bar, fig_pie
    
    def get_imbalance_statistics(self):
        """Generate detailed imbalance statistics"""
        if self.target_col is None:
            return None, None
        
        imbalance = self.analyze_class_imbalance()
        if imbalance is None:
            return None, None
        
        stats_df = pd.DataFrame({
            'Clase': imbalance['class_counts'].index,
            'Frecuencia': imbalance['class_counts'].values,
            'Porcentaje': imbalance['class_percentages'].values
        })
        
        # Add recommendations
        recommendations = []
        if imbalance['is_imbalanced']:
            recommendations.append("⚠️ Desequilibrio detectado")
            recommendations.append(f"📊 Ratio: {imbalance['imbalance_ratio']:.2f}:1")
            recommendations.append("💡 Recomendaciones para manejar el desequilibrio:")
            recommendations.append("   • SMOTE (Synthetic Minority Oversampling)")
            recommendations.append("   • Random Oversampling de clase minoritaria")
            recommendations.append("   • Random Undersampling de clase mayoritaria")
            recommendations.append("   • Weighted Loss Functions")
            recommendations.append("   • Stratified Cross-Validation")
            recommendations.append("   • Threshold Tuning")
        else:
            recommendations.append("✅ Clases balanceadas")
        
        return stats_df, recommendations
    
    def get_missing_data_analysis(self):
        """Analyze missing values in the dataset"""
        missing_data = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum(),
            'Missing_Percentage': (self.df.isnull().sum() / len(self.df) * 100).round(2)
        })
        
        missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values(
            'Missing_Percentage', ascending=False
        )
        
        return missing_data
    
    def plot_missing_data(self):
        """Visualize missing data"""
        missing_data = self.get_missing_data_analysis()
        
        if len(missing_data) == 0:
            return None
        
        fig = px.bar(
            missing_data,
            x='Column',
            y='Missing_Percentage',
            title='Análisis de Datos Faltantes (%)',
            labels={'Column': 'Variable', 'Missing_Percentage': 'Porcentaje Faltante'},
            color='Missing_Percentage',
            color_continuous_scale='Reds'
        )
        
        return fig
    
    def detect_outliers(self, method='iqr', threshold=1.5):
        """Detect outliers using IQR method"""
        outliers_info = {}
        
        for col in self.numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                outliers_info[col] = {
                    'count': outlier_count,
                    'percentage': (outlier_count / len(self.df) * 100).round(2),
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
        
        return pd.DataFrame(outliers_info).T if outliers_info else pd.DataFrame()
    
    def plot_outliers(self):
        """Visualize outliers detection"""
        outliers_df = self.detect_outliers()
        
        if len(outliers_df) == 0:
            return None
        
        fig = px.bar(
            outliers_df.reset_index(),
            x='index',
            y='percentage',
            title='Detección de Outliers por Variable (%)',
            labels={'index': 'Variable', 'percentage': 'Porcentaje de Outliers'},
            color='percentage',
            color_continuous_scale='Oranges'
        )
        
        return fig
    
    def feature_importance_with_target(self):
        """Calculate feature importance relative to target variable"""
        if self.target_col is None or self.target_col not in self.df.columns:
            return None
        
        importance_scores = []
        
        # Encode target if categorical
        target_data = self.df[self.target_col]
        if target_data.dtype == 'object':
            le = LabelEncoder()
            target_encoded = le.fit_transform(target_data)
        else:
            target_encoded = target_data
        
        # Calculate correlation with target for numeric features
        for col in self.numeric_cols:
            if col != self.target_col:
                corr, p_value = pointbiserialr(target_encoded, self.df[col])
                importance_scores.append({
                    'Feature': col,
                    'Correlation': abs(corr),
                    'P_Value': p_value
                })
        
        importance_df = pd.DataFrame(importance_scores)
        importance_df = importance_df.sort_values('Correlation', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        importance_df = self.feature_importance_with_target()
        
        if importance_df is None or len(importance_df) == 0:
            return None
        
        fig = px.bar(
            importance_df.head(15),
            x='Correlation',
            y='Feature',
            orientation='h',
            title='Top 15 Features - Correlación con Target',
            labels={'Correlation': 'Correlación Absoluta', 'Feature': 'Característica'},
            color='Correlation',
            color_continuous_scale='Viridis'
        )
        
        return fig


if __name__ == "__main__":
    print("✅ Feature Engineering Pipeline module loaded successfully")
