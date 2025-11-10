import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from IPython.display import display, HTML, clear_output
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets

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
    def __init__(self, figsize: tuple = (10, 6)):
        self.figsize = figsize
        plt.style.use('default')
    
    def _create_tabs(self, titles):
        """Create collapsible tabs for organized output"""
        tab_titles = titles
        children = [widgets.Output() for _ in tab_titles]
        tab = widgets.Tab(children=children)
        for i, title in enumerate(tab_titles):
            tab.set_title(i, title)
        return tab, children
    
    def quick_eda(self, X: pd.DataFrame, y: Optional[pd.Series] = None, max_categories: int = 10):
        """
        All-in-one EDA with collapsible sections
        """
        # Create tabs for different sections
        tab_titles = ['Overview', 'Numeric Features', 'Categorical Features', 'Missing Data', 'Correlation Heatmap']
        if y is not None:
            tab_titles.append('Target Relationships')
            
        tab, outputs = self._create_tabs(tab_titles)
        display(tab)
        
        # Tab 1: Overview
        with outputs[0]:
            clear_output()
            self._show_overview(X, y)
        
        # Tab 2: Numeric Features
        with outputs[1]:
            clear_output()
            self.plot_numeric_distributions(X)
        
        # Tab 3: Categorical Features  
        with outputs[2]:
            clear_output()
            self.plot_categorical_counts(X, top_n=max_categories)
        
        # Tab 4: Missing Data
        with outputs[3]:
            clear_output()
            self.plot_missing_values(X)
        
        # Tab 5: Correlation Heatmap
        with outputs[4]:
            clear_output()
            self.plot_correlation_heatmap(X)
        
        # Tab 6: Target Relationships
        if y is not None and len(outputs) > 5:
            with outputs[5]:
                clear_output()
                self.plot_target_relationships(X, y)
    
    def _show_overview(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Show dataframe overview in compact format"""
        overview_html = f"""
        <div style="font-family: Arial; font-size: 12px;">
            <h3>Dataset Overview</h3>
            <p><b>Shape:</b> {X.shape[0]:,} rows × {X.shape[1]:,} columns</p>
            <p><b>Numeric Features:</b> {len(X.select_dtypes(include='number').columns)}</p>
            <p><b>Categorical Features:</b> {len(X.select_dtypes(include=['object', 'category']).columns)}</p>
            <p><b>Memory Usage:</b> {X.memory_usage(deep=True).sum() / 1024**2:.1f} MB</p>
        """
        if y is not None:
            overview_html += f"""
            <p><b>Target Variable:</b> {y.name if hasattr(y, 'name') else 'Target'}</p>
            <p><b>Target Distribution:</b><br>{y.value_counts().to_frame().to_html(classes='table table-striped', header=False)}</p>
            """
        overview_html += "</div>"
        display(HTML(overview_html))
    
    def plot_numeric_distributions(self, X: pd.DataFrame, cols: Optional[List[str]] = None):
        """Compact numeric distributions in grid layout"""
        if cols is None:
            cols = X.select_dtypes(include=['number']).columns.tolist()
        
        if not cols:
            display(HTML("<p>No numeric columns found</p>"))
            return
        
        n_cols = min(3, len(cols))
        n_rows = (len(cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0], 2.5*n_rows))
        if n_rows * n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        for i, col in enumerate(cols):
            if i < len(axes):
                # Histogram with KDE
                X[col].hist(bins=30, ax=axes[i], alpha=0.7, density=True, color='skyblue')
                X[col].plot.density(ax=axes[i], color='red', linewidth=2)
                axes[i].set_title(f'{col}\n(μ={X[col].mean():.2f}, σ={X[col].std():.2f})', fontsize=10)
                axes[i].tick_params(labelsize=8)
        
        # Hide empty subplots
        for i in range(len(cols), len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.show()
    
    def plot_categorical_counts(self, X: pd.DataFrame, cols: Optional[List[str]] = None, top_n: int = 10):
        """Compact categorical value counts"""
        if cols is None:
            cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not cols:
            display(HTML("<p>No categorical columns found</p>"))
            return
        
        n_cols = min(2, len(cols))
        n_rows = (len(cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0], 3*n_rows))
        if n_rows * n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        for i, col in enumerate(cols):
            if i < len(axes):
                value_counts = X[col].value_counts().head(top_n)
                bars = axes[i].bar(range(len(value_counts)), value_counts.values, color='lightcoral', alpha=0.7)
                axes[i].set_title(f'{col}\n({X[col].nunique()} unique)', fontsize=10)
                axes[i].set_xticks(range(len(value_counts)))
                axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right', fontsize=8)
                axes[i].tick_params(labelsize=8)
                
                # Add value labels on bars
                for bar, count in zip(bars, value_counts.values):
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height,
                                f'{count}', ha='center', va='bottom', fontsize=8)
        
        # Hide empty subplots
        for i in range(len(cols), len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.show()
    
    def plot_missing_values(self, X: pd.DataFrame):
        """Compact missing values visualization"""
        missing_count = X.isnull().sum()
        missing_pct = (missing_count / len(X)) * 100
        missing_data = pd.DataFrame({
            'Column': missing_count.index,
            'Missing_Count': missing_count.values,
            'Missing_Percent': missing_pct.values
        }).sort_values('Missing_Percent', ascending=False)
        missing_data = missing_data[missing_data['Missing_Count'] > 0]
        
        if missing_data.empty:
            display(HTML("<p>No missing values found</p>"))
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Count plot
        bars1 = ax1.barh(missing_data['Column'], missing_data['Missing_Count'], color='salmon')
        ax1.set_title('Missing Values Count')
        ax1.set_xlabel('Number of Missing Values')
        
        # Percentage plot
        bars2 = ax2.barh(missing_data['Column'], missing_data['Missing_Percent'], color='lightcoral')
        ax2.set_title('Missing Values Percentage')
        ax2.set_xlabel('Percentage Missing (%)')
        ax2.set_xlim(0, 100)
        
        # Add value annotations
        for bar in bars1:
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2., f'{int(width)}', 
                    ha='left', va='center', fontsize=8)
        
        for bar in bars2:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2., f'{width:.1f}%', 
                    ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    def plot_target_relationships(self, X: pd.DataFrame, y: pd.Series, max_features: int = 6):
        """Compact target relationships"""
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            display(HTML("<p>No numeric features for target relationship analysis</p>"))
            return
        
        # Select top correlated features
        correlations = X[numeric_cols].corrwith(y).abs().sort_values(ascending=False)
        top_features = correlations.head(max_features).index.tolist()
        
        n_cols = min(3, len(top_features))
        n_rows = (len(top_features) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
        if n_rows * n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        for i, col in enumerate(top_features):
            if i < len(axes):
                scatter = axes[i].scatter(X[col], y, alpha=0.5, s=20)
                axes[i].set_xlabel(col, fontsize=9)
                axes[i].set_ylabel('Target', fontsize=9)
                corr = correlations[col]
                axes[i].set_title(f'{col}\n(corr: {corr:.3f})', fontsize=10)
                axes[i].tick_params(labelsize=8)
        
        # Hide empty subplots
        for i in range(len(top_features), len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_heatmap(self, X: pd.DataFrame, method: str = 'pearson', figsize: tuple = (12, 10)):
        """Plot correlation heatmap for numeric features"""
        numeric_cols = X.select_dtypes(include=['number']).columns
        if len(numeric_cols) < 2:
            display(HTML("<p>Need at least 2 numeric columns for correlation heatmap</p>"))
            return
        
        # Calculate correlation matrix
        corr_matrix = X[numeric_cols].corr(method=method)
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Plot heatmap
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   annot=True, 
                   fmt='.2f', 
                   cmap='RdBu_r', 
                   center=0,
                   square=True, 
                   linewidths=0.5,
                   cbar_kws={'shrink': 0.8})
        
        plt.title(f'Feature Correlation Matrix ({method.capitalize()})', fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        # Display high correlation pairs
        self._show_high_correlations(corr_matrix)
    
    def _show_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7):
        """Display highly correlated feature pairs"""
        # Get upper triangle of correlation matrix without diagonal
        corr_upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
        
        # Find high correlations
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
            high_corr_df = pd.DataFrame(high_corr_pairs)
            display(HTML(f"<h4>High Correlations (>{threshold}):</h4>"))
            display(high_corr_df.style.set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#f0f0f0'), 
                                           ('font-weight', 'bold')]}
            ]))
