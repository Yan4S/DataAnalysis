import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from IPython.display import display, HTML, clear_output
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
import logging
import warnings

logging.basicConfig(level=logging.INFO, format='%(message)s')
warnings.filterwarnings('ignore')


def typeConversion(_df, convert_to_cat, convert_to_num):
    df = _df.copy()
    
    # Convert numeric to categorical (column by column)
    for col in convert_to_cat:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Convert categorical to numeric (column by column)
    for col in convert_to_num:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes
    
    return df
    
class FeatureTypeAnalyzer:
    def __init__(self):
        self.suspicious_features = {}
    
    def analyze_suspicious_features(self, X: pd.DataFrame) -> Dict:
        """
        Analyze and explain why features might need type conversion
        """
        self.suspicious_features = {
            'numeric_that_should_be_categorical': [],
            'categorical_that_should_be_numeric': []
        }
        
        # Analyze numeric columns that might actually be categorical
        numeric_cols = X.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            reasons = self._analyze_numeric_suspicious(X, col)
            if reasons:
                self.suspicious_features['numeric_that_should_be_categorical'].append({
                    'column': col,
                    'reasons': reasons,
                    'stats': {
                        'unique_values': X[col].nunique(),
                        'unique_ratio': X[col].nunique() / len(X),
                        'actual_unique_values': sorted(X[col].dropna().unique()) if X[col].nunique() <= 20 else 'Too many to show'
                    }
                })
        
        # Analyze categorical columns that might actually be numeric  
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            reasons = self._analyze_categorical_suspicious(X, col)
            if reasons:
                self.suspicious_features['categorical_that_should_be_numeric'].append({
                    'column': col,
                    'reasons': reasons,
                    'stats': {
                        'unique_values': X[col].nunique(),
                        'actual_unique_values': sorted(X[col].dropna().unique()) if X[col].nunique() <= 20 else 'Too many to show',
                        'data_type': str(X[col].dtype)
                    }
                })
        
        return self.suspicious_features

      
    def _analyze_numeric_suspicious(self, X: pd.DataFrame, col: str) -> List[str]:
        """Explain why a numeric column might be categorical"""
        reasons = []
        
        # Low cardinality check
        unique_count = X[col].nunique()
        unique_ratio = unique_count / len(X)
        
        if unique_count <= 15:
            reasons.append(f"Only {unique_count} unique values ({unique_ratio:.1%} of data)")
        
        # Integer values with small range (often categorical)
        if X[col].dtype in ['int64', 'int32']:
            value_range = X[col].max() - X[col].min()
            if value_range <= 20 and unique_count <= 15:
                reasons.append(f"Small integer range ({value_range} values) - typical for categories")
        
        # All values are non-negative integers (common for IDs, codes)
        if (X[col] >= 0).all() and (X[col] == X[col].astype(int)).all():
            if unique_ratio > 0.95:  # Mostly unique values
                reasons.append("All unique integer values - likely an ID/code")
            elif unique_count <= 50:
                reasons.append("Non-negative integers with low cardinality - likely categorical codes")
        
        # Check if name suggests categorical nature
        categorical_keywords = ['id', 'ID', 'Id', 'code', 'Code', 'type', 'Type', 'level', 'Level', 'flag', 'Flag', 'group', 'Group']
        if any(keyword in col.lower() for keyword in categorical_keywords):
            reasons.append(f"Column name suggests categorical nature")
        
        return reasons
    def _analyze_categorical_suspicious(self, X: pd.DataFrame, col: str) -> List[str]:
        """Explain why a categorical column might be numeric"""
        reasons = []
        
        # Try to convert to numeric
        numeric_vals = pd.to_numeric(X[col], errors='coerce')
        non_null_count = numeric_vals.notna().sum()
        conversion_rate = non_null_count / len(X)
        
        if conversion_rate > 0.8:
            reasons.append(f"{conversion_rate:.1%} of values can be converted to numbers")
            
            # Check if it's ordinal data
            unique_vals = X[col].dropna().unique()
            if len(unique_vals) <= 10:
                ordinal_patterns = [
                    {'low', 'medium', 'high'},
                    {'small', 'medium', 'large'}, 
                    {'bad', 'good', 'excellent'},
                    {'1', '2', '3', '4', '5'}
                ]
                current_vals = set(str(v).lower() for v in unique_vals)
                if any(pattern.issubset(current_vals) for pattern in ordinal_patterns):
                    reasons.append("Values follow ordinal pattern")
        
        # Check if name suggests numeric nature
        numeric_keywords = ['score', 'rating','rank', 'level', 'priority', 'year', 'yr', 'time', 'mo', 'min']
        if any(keyword.lower() in col.lower() for keyword in numeric_keywords):
            reasons.append(f"Column name suggests numeric/ordinal nature")
        
        # Check if values look like numbers stored as strings
        sample_values = X[col].dropna().head(10)
        numeric_looking = sum(str(val).replace('.', '').replace('-', '').isdigit() for val in sample_values)
        if numeric_looking / len(sample_values) > 0.7:
            reasons.append("Most values look like numbers stored as text")
        
        return reasons
    def display_analysis_report(self, X: pd.DataFrame):
        """Show interactive analysis report with explanations"""
        self.analyze_suspicious_features(X)
        
        display(HTML("<h3>Feature Type Analysis Report</h3>"))
        
        # Numeric that should be categorical
        if self.suspicious_features['numeric_that_should_be_categorical']:
            display(HTML("<h4>Numeric Columns That Might Be Categorical:</h4>"))
            for feature in self.suspicious_features['numeric_that_should_be_categorical']:
                unique_vals = feature['stats']['actual_unique_values']
                if unique_vals != 'Too many to show':
                    unique_str = f"Values: {unique_vals}"
                else:
                    unique_str = f"Too many unique values to display ({feature['stats']['unique_values']} total)"
                
                display(HTML(f"""
                <div style="border-left: 4px solid #ff6b6b; padding-left: 10px; margin: 10px 0;">
                    <b>{feature['column']}</b><br>
                    <small>
                    Unique: {feature['stats']['unique_values']} values | 
                    Ratio: {feature['stats']['unique_ratio']:.1%}<br>
                    {unique_str}
                    </small><br>
                    Reasons: {', '.join(feature['reasons'])}
                </div>
                """))
        
        # Categorical that should be numeric
        if self.suspicious_features['categorical_that_should_be_numeric']:
            display(HTML("<h4>Categorical Columns That Might Be Numeric:</h4>"))
            for feature in self.suspicious_features['categorical_that_should_be_numeric']:
                unique_vals = feature['stats']['actual_unique_values']
                if unique_vals != 'Too many to show':
                    unique_str = f"Values: {unique_vals}"
                else:
                    unique_str = f"Too many unique values to display ({feature['stats']['unique_values']} total)"
                
                display(HTML(f"""
                <div style="border-left: 4px solid #4ecdc4; padding-left: 10px; margin: 10px 0;">
                    <b>{feature['column']}</b><br>
                    <small>
                    Unique: {feature['stats']['unique_values']} values | 
                    Type: {feature['stats']['data_type']}<br>
                    {unique_str}
                    </small><br>
                    Reasons: {', '.join(feature['reasons'])}
                </div>
                """))
        
        if not any(self.suspicious_features.values()):
            display(HTML("<p>No suspicious feature types detected - all types look appropriate!</p>"))
    
    def get_conversion_recommendations(self) -> Dict:
        """Get specific conversion recommendations"""
        recommendations = {
            'convert_to_categorical': [],
            'convert_to_numeric': []
        }
        
        for feature in self.suspicious_features['numeric_that_should_be_categorical']:
            recommendations['convert_to_categorical'].append(feature['column'])
        
        for feature in self.suspicious_features['categorical_that_should_be_numeric']:
            recommendations['convert_to_numeric'].append(feature['column'])
        
        return recommendations



class DataPlotter:
    """
    Simple plotting class - you control all parameters directly
    """
    
    def __init__(self, figsize: tuple = (10, 6)):
        self.figsize = figsize
        plt.style.use('default')
        
        logging.info("SIMPLE DATA PLOTTER INITIALIZED")
        logging.info("=" * 50)
        logging.info("USAGE EXAMPLES:")
        logging.info("=" * 50)
        
        logging.info("\n1. Plot numeric features:")
        logging.info("   plotter.plot_numeric_features(X, plot_type='histogram', ncols=3)")
        logging.info("   plotter.plot_numeric_features(X, plot_type='boxplot', col_names=['age', 'income'])")
        
        logging.info("\n2. Plot categorical features:")
        logging.info("   plotter.plot_categorical_features(X, max_categories=10, ncols=2)")
        logging.info("   plotter.plot_categorical_features(X, col_names=['category', 'city'])")
        
        logging.info("\n3. Plot missing values:")
        logging.info("   plotter.plot_missing_values(X)")
        logging.info("   plotter.plot_missing_values(X, col_names=['col1', 'col2', 'col3'])")
        
        logging.info("\n4. Plot correlations:")
        logging.info("   plotter.plot_correlations(X, n_features=10, method='pearson')")
        logging.info("   plotter.plot_correlations(X, n_features='all', threshold=0.8)")
        
        logging.info("\n5. Plot target relationships:")
        logging.info("   plotter.plot_target_relationships(X, y, plot_type='scatter')")
        logging.info("   plotter.plot_target_relationships(X, y, n_features=5, col_names=['feat1', 'feat2'])")
        
        logging.info("\n" + "=" * 50)
        logging.info("TIP: Call any plotting function to see its specific options!")
        logging.info("=" * 50)
    
    def plot_numeric_features(self, X: pd.DataFrame, plot_type: str = 'histogram', ncols: int = 5, col_names: Optional[List[str]] = None):
        """Plot numeric features with specified type"""
        logging.info("NUMERIC FEATURES PLOT OPTIONS:")
        logging.info("plot_type: 'histogram', 'boxplot', 'density', 'violin'")
        logging.info("ncols: 2-4 (suggested: 3)")
        logging.info("col_names: list of column names or None for all numeric columns")
        logging.info(f"Your choice: plot_type='{plot_type}', ncols={ncols}, col_names={col_names}")
        
        if col_names is None:
            numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        else:
            numeric_cols = [col for col in col_names if col in X.columns and pd.api.types.is_numeric_dtype(X[col])]
        
        if not numeric_cols:
            logging.info("No numeric columns found")
            return
        
        n_features = len(numeric_cols)
        nrows = (n_features + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
        axes_flat = axes.flatten() if nrows * ncols > 1 else [axes]
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes_flat):
                ax = axes_flat[i]
                
                if plot_type == 'histogram':
                    sns.histplot(data=X, x=col, ax=ax, kde=True)
                    ax.set_title(f'{col}')
                elif plot_type == 'boxplot':
                    sns.boxplot(data=X, y=col, ax=ax)
                    ax.set_title(f'{col}')
                elif plot_type == 'density':
                    sns.kdeplot(data=X, x=col, ax=ax, fill=True)
                    ax.set_title(f'{col}')
                elif plot_type == 'violin':
                    sns.violinplot(data=X, y=col, ax=ax)
                    ax.set_title(f'{col}')
        
        # Hide empty subplots
        for i in range(len(numeric_cols), len(axes_flat)):
            axes_flat[i].set_visible(False)
            
        plt.tight_layout()
        plt.show()
    
    def plot_categorical_features(self, X: pd.DataFrame, max_categories: int = 10, ncols: int = 5, col_names: Optional[List[str]] = None):
        """Plot categorical features"""
        logging.info("CATEGORICAL FEATURES PLOT OPTIONS:")
        logging.info("max_categories: 5-20 (suggested: 10)")
        logging.info("ncols: 2-3 (suggested: 2)")
        logging.info("col_names: list of column names or None for all categorical columns")
        logging.info(f"Your choice: max_categories={max_categories}, ncols={ncols}, col_names={col_names}")
        
        if col_names is None:
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            categorical_cols = [col for col in col_names if col in X.columns and not pd.api.types.is_numeric_dtype(X[col])]
        
        if not categorical_cols:
            logging.info("No categorical columns found")
            return
        
        n_features = len(categorical_cols)
        nrows = (n_features + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4))
        axes_flat = axes.flatten() if nrows * ncols > 1 else [axes]
        
        for i, col in enumerate(categorical_cols):
            if i < len(axes_flat):
                ax = axes_flat[i]
                value_counts = X[col].value_counts().head(max_categories)
                
                bars = ax.bar(range(len(value_counts)), value_counts.values, color='lightcoral', alpha=0.7)
                ax.set_title(f'{col}\n({X[col].nunique()} unique)')
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
        
        # Hide empty subplots
        for i in range(len(categorical_cols), len(axes_flat)):
            axes_flat[i].set_visible(False)
            
        plt.tight_layout()
        plt.show()
    
    def plot_missing_values(self, X: pd.DataFrame, col_names: Optional[List[str]] = None):
        """Plot missing values"""
        logging.info("MISSING VALUES PLOT:")
        logging.info("col_names: list of column names or None for all columns")
        logging.info(f"Your choice: col_names={col_names}")
        
        if col_names is None:
            data_to_plot = X
        else:
            data_to_plot = X[col_names]
        
        missing_count = data_to_plot.isnull().sum()
        missing_pct = (missing_count / len(data_to_plot)) * 100
        missing_data = pd.DataFrame({
            'Column': missing_count.index,
            'Missing_Count': missing_count.values,
            'Missing_Percent': missing_pct.values
        }).sort_values('Missing_Percent', ascending=False)
        missing_data = missing_data[missing_data['Missing_Count'] > 0]
        
        if missing_data.empty:
            logging.info("No missing values found")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Count plot
        ax1.barh(missing_data['Column'], missing_data['Missing_Count'], color='salmon')
        ax1.set_title('Missing Values Count')
        
        # Percentage plot
        ax2.barh(missing_data['Column'], missing_data['Missing_Percent'], color='lightcoral')
        ax2.set_title('Missing Values Percentage')
        ax2.set_xlim(0, 100)
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlations(self, X: pd.DataFrame, n_features: int = 10, method: str = 'pearson', threshold: float = 0.7, col_names: Optional[List[str]] = None):
        """Plot correlation heatmap"""
        logging.info("CORRELATION PLOT OPTIONS:")
        logging.info("n_features: 5-20 or use 'all' for all columns")
        logging.info("method: 'pearson', 'spearman'")
        logging.info("threshold: 0.5-0.9 (suggested: 0.7)")
        logging.info("col_names: list of column names or None for all numeric columns")
        logging.info(f"Your choice: n_features={n_features}, method='{method}', threshold={threshold}, col_names={col_names}")
        
        if col_names is None:
            numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        else:
            numeric_cols = [col for col in col_names if col in X.columns and pd.api.types.is_numeric_dtype(X[col])]
        
        if len(numeric_cols) < 2:
            logging.info("Need at least 2 numeric columns for correlation analysis")
            return
        
        # Select features
        if n_features == 'all' or n_features >= len(numeric_cols):
            selected_cols = numeric_cols
        else:
            variances = X[numeric_cols].var().sort_values(ascending=False)
            selected_cols = variances.head(n_features).index.tolist()
        
        corr_matrix = X[selected_cols].corr(method=method)
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
        plt.title(f'Correlation Matrix ({method.capitalize()})')
        plt.tight_layout()
        plt.show()
        
        # Show high correlations
        self._show_high_correlations(corr_matrix, threshold)
    
    def plot_target_relationships(self, X: pd.DataFrame, y: pd.Series, n_features: int = 10, plot_type: str = 'scatter', col_names: Optional[List[str]] = None):
        """Plot relationships between features and target"""
        logging.info("TARGET RELATIONSHIPS PLOT OPTIONS:")
        logging.info("n_features: 5-15 (suggested: 10)")
        logging.info("plot_type: 'scatter', 'boxplot', 'violin', 'stripplot'")
        logging.info("col_names: list of column names or None for all numeric columns")
        logging.info(f"Your choice: n_features={n_features}, plot_type='{plot_type}', col_names={col_names}")
        
        if col_names is None:
            numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        else:
            numeric_cols = [col for col in col_names if col in X.columns and pd.api.types.is_numeric_dtype(X[col])]
        
        if not numeric_cols:
            logging.info("No numeric columns found")
            return
        
        # Select top correlated features with target
        correlations = X[numeric_cols].corrwith(y).abs().sort_values(ascending=False)
        top_features = correlations.head(n_features).index.tolist()
        
        ncols = 2
        nrows = (len(top_features) + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes_flat = axes.flatten() if nrows * ncols > 1 else [axes]
        
        for i, col in enumerate(top_features):
            if i < len(axes_flat):
                ax = axes_flat[i]
                
                if plot_type == 'scatter':
                    ax.scatter(X[col], y, alpha=0.5, s=20)
                    ax.set_xlabel(col)
                    ax.set_ylabel('Target')
                elif plot_type == 'boxplot':
                    plot_data = pd.DataFrame({col: X[col], 'Target': y})
                    sns.boxplot(data=plot_data, x='Target', y=col, ax=ax)
                
                corr = correlations[col]
                ax.set_title(f'{col}\n(corr: {corr:.3f})')
        
        # Hide empty subplots
        for i in range(len(top_features), len(axes_flat)):
            axes_flat[i].set_visible(False)
            
        plt.tight_layout()
        plt.show()
    
    def _show_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float):
        """Display highly correlated feature pairs"""
        corr_upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
        
        high_corr_pairs = []
        for col in corr_upper.columns:
            high_corrs = corr_upper[col][abs(corr_upper[col]) > threshold]
            for feature, corr in high_corrs.items():
                high_corr_pairs.append({
                    'Feature 1': col,
                    'Feature 2': feature,
                    'Correlation': f'{corr:.3f}'
                })
        
        if high_corr_pairs:
            logging.info(f"High correlations (> {threshold}):")
            high_corr_df = pd.DataFrame(high_corr_pairs)
            print(high_corr_df.to_string(index=False))
        else:
            logging.info(f"No correlations above {threshold} found")

