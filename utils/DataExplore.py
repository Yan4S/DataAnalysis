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
    
    
    def interactive_eda(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Interactive EDA with controls for each visualization type
        """
        # Create main tabs
        tab_titles = ['Overview', 'Numeric Features', 'Categorical Features', 
                     'Missing Data', 'Correlations', 'Target Analysis']
        tab, outputs = self._create_tabs(tab_titles)
        display(tab)
        
        # Tab 1: Overview
        with outputs[0]:
            clear_output()
            self._show_overview(X, y)
        
        # Tab 2: Numeric Features with controls
        with outputs[1]:
            clear_output(wait=True)
            self._setup_numeric_controls(X)
        
        # Tab 3: Categorical Features with controls  
        with outputs[2]:
            clear_output(wait=True)
            self._setup_categorical_controls(X)
        
        # Tab 4: Missing Data
        with outputs[3]:
            clear_output()
            self.plot_missing_values(X)
        
        # Tab 5: Correlations with controls
        with outputs[4]:
            clear_output(wait=True)
            self._setup_correlation_controls(X)
        
        # Tab 6: Target Analysis with controls
        with outputs[5]:
            clear_output(wait=True)
            if y is not None:
                self._setup_target_controls(X, y)
            else:
                display(HTML("<p>No target variable provided for analysis</p>"))
    
    def _show_overview(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Show dataframe overview"""
        overview_html = f"""
        <div style="font-family: Arial; font-size: 14px;">
            <h3>Dataset Overview</h3>
            <p><b>Shape:</b> {X.shape[0]:,} rows × {X.shape[1]:,} columns</p>
            <p><b>Numeric Features:</b> {len(X.select_dtypes(include='number').columns)}</p>
            <p><b>Categorical Features:</b> {len(X.select_dtypes(include=['object', 'category']).columns)}</p>
            <p><b>Memory Usage:</b> {X.memory_usage(deep=True).sum() / 1024**2:.1f} MB</p>
        """
        if y is not None:
            target_info = f"""
            <p><b>Target Variable:</b> {y.name if hasattr(y, 'name') else 'Target'}</p>
            <p><b>Target Type:</b> {'Numeric' if pd.api.types.is_numeric_dtype(y) else 'Categorical'}</p>
            <p><b>Unique Values:</b> {y.nunique()}</p>
            """
            overview_html += target_info
        overview_html += "</div>"
        display(HTML(overview_html))
    
    def _setup_numeric_controls(self, X: pd.DataFrame):
        """Setup interactive controls for numeric features"""
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            display(HTML("<p>No numeric columns found</p>"))
            return
        
        # Controls
        plot_type = widgets.Dropdown(
            options=['Histogram', 'Boxplot', 'Density', 'Violin'],
            value='Histogram',
            description='Plot Type:',
            style={'description_width': 'initial'}
        )
        
        columns_per_row = widgets.Dropdown(
            options=[3, 4, 5, 6],
            value=5,
            description='Columns per Row:',
            style={'description_width': 'initial'}
        )
        
        update_num_btn = widgets.Button(description='Update Plot', button_style='primary')
        output = widgets.Output()
        
        def on_update_clicked(b):
            with output:
                clear_output(wait=True)
                self._plot_numeric_features(X, plot_type.value, columns_per_row.value)
        
        update_num_btn.on_click(on_update_clicked)
        
        # Display controls
        display(HTML("<h3>Numeric Features Analysis</h3>"))
        display(widgets.HBox([plot_type, columns_per_row]))
        display(update_num_btn)
        display(output)
        
        # Initial plot
        with output:
            self._plot_numeric_features(X, plot_type.value, columns_per_row.value)
    
    def _plot_numeric_features(self, X: pd.DataFrame, plot_type: str, ncols: int):
        """Plot numeric features based on selected type"""
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        n_features = len(numeric_cols)
        nrows = (n_features + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
        axes_flat = axes.flatten() if nrows * ncols > 1 else [axes]
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes_flat):
                ax = axes_flat[i]
                
                if plot_type == 'Histogram':
                    sns.histplot(data=X, x=col, ax=ax, kde=True)
                    ax.set_title(f'{col}\n(μ={X[col].mean():.2f}, σ={X[col].std():.2f})')
                elif plot_type == 'Boxplot':
                    sns.boxplot(data=X, y=col, ax=ax)
                    ax.set_title(f'{col}')
                elif plot_type == 'Density':
                    sns.kdeplot(data=X, x=col, ax=ax, fill=True)
                    ax.set_title(f'{col}')
                elif plot_type == 'Violin':
                    sns.violinplot(data=X, y=col, ax=ax)
                    ax.set_title(f'{col}')
                
                ax.tick_params(labelsize=9)
        
        # Hide empty subplots
        for i in range(len(numeric_cols), len(axes_flat)):
            axes_flat[i].set_visible(False)
            
        plt.tight_layout()
        plt.show()
    
    def _setup_categorical_controls(self, X: pd.DataFrame):
        """Setup interactive controls for categorical features"""
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_cols:
            display(HTML("<p>No categorical columns found</p>"))
            return
        
        # Controls
        max_categories = widgets.Dropdown(
            options=[5, 10, 15, 20, 25],
            value=15,
            description='Max Categories:',
            style={'description_width': 'initial'}
        )
        
        columns_per_row = widgets.Dropdown(
            options=[3, 4, 5, 6],
            value=5,
            description='Columns per Row:',
            style={'description_width': 'initial'}
        )
        
        update_cat_btn = widgets.Button(description='Update Plot', button_style='primary')
        output = widgets.Output()
        
        def on_update_clicked(b):
            with output:
                clear_output(wait=True)
                self._plot_categorical_features(X, max_categories.value, columns_per_row.value)
        
        update_cat_btn.on_click(on_update_clicked)
        
        # Display controls
        display(HTML("<h3>Categorical Features Analysis</h3>"))
        display(widgets.HBox([max_categories, columns_per_row]))
        display(update_cat_btn)
        display(output)
        
        # Initial plot
        with output:
            self._plot_categorical_features(X, max_categories.value, columns_per_row.value)
        on_update_clicked(None)
    
    def _plot_categorical_features(self, X: pd.DataFrame, max_cats: int, ncols: int):
        """Plot categorical features"""
        categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()
        n_features = len(categorical_cols)
        nrows = (n_features + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
        axes_flat = axes.flatten() if nrows * ncols > 1 else [axes]
        
        for i, col in enumerate(categorical_cols):
            if i < len(axes_flat):
                ax = axes_flat[i]
                value_counts = X[col].value_counts().head(max_cats)
                
                bars = ax.bar(range(len(value_counts)), value_counts.values, 
                            color='lightcoral', alpha=0.7)
                ax.set_title(f'{col}\n({X[col].nunique()} unique)', fontsize=12)
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index, rotation=45, ha='right', fontsize=9)
                
                # Add value labels on bars
                for bar, count in zip(bars, value_counts.values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{count}', ha='center', va='bottom', fontsize=9)
        
        # Hide empty subplots
        for i in range(len(categorical_cols), len(axes_flat)):
            axes_flat[i].set_visible(False)
            
        plt.tight_layout()
        plt.show()
    
    def plot_missing_values(self, X: pd.DataFrame):
        """Missing values visualization"""
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
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Count plot
        bars1 = ax1.barh(missing_data['Column'], missing_data['Missing_Count'], color='salmon')
        ax1.set_title('Missing Values Count', fontsize=14)
        ax1.set_xlabel('Number of Missing Values')
        
        # Percentage plot
        bars2 = ax2.barh(missing_data['Column'], missing_data['Missing_Percent'], color='lightcoral')
        ax2.set_title('Missing Values Percentage', fontsize=14)
        ax2.set_xlabel('Percentage Missing (%)')
        ax2.set_xlim(0, 100)
        
        # Add value annotations
        for bar in bars1:
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2., f'{int(width)}', 
                    ha='left', va='center', fontsize=10)
        
        for bar in bars2:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2., f'{width:.1f}%', 
                    ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def _setup_correlation_controls(self, X: pd.DataFrame):
        """Setup interactive controls for correlation analysis"""
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 2:
            display(HTML("<p>Need at least 2 numeric columns for correlation analysis</p>"))
            return
        
        # Controls
        n_features = widgets.Dropdown(
            options=[5, 10, 15, 20, 'All'],
            value=15,
            description='Top Features:',
            style={'description_width': 'initial'}
        )
        
        correlation_type = widgets.Dropdown(
            options=['Pearson', 'Spearman'],
            value='Pearson',
            description='Method:',
            style={'description_width': 'initial'}
        )
        
        threshold = widgets.Dropdown(
            options=[0.5, 0.6, 0.7, 0.8, 0.9],
            value=0.7,
            description='High Corr Threshold:',
            style={'description_width': 'initial'}
        )
        
        update_corr_btn = widgets.Button(description='Update Heatmap', button_style='primary')
        output = widgets.Output()
        
        def on_update_clicked(b):
            with output:
                clear_output(wait=True)
                n_feats = len(numeric_cols) if n_features.value == 'All' else n_features.value
                method = correlation_type.value.lower()
                self._plot_correlation_heatmap(X, n_feats, method, threshold.value)
        
        update_corr_btn.on_click(on_update_clicked)
        
        # Display controls
        display(HTML("<h3>Feature Correlation Analysis</h3>"))
        display(widgets.HBox([n_features, correlation_type, threshold]))
        display(update_corr_btn)
        display(output)
        
        # Initial plot
        with output:
            n_feats = len(numeric_cols) if n_features.value == 'All' else n_features.value
            self._plot_correlation_heatmap(X, n_feats, correlation_type.value.lower(), threshold.value)
        on_update_clicked(None)
    
    def _plot_correlation_heatmap(self, X: pd.DataFrame, n_features: int, method: str, threshold: float):
        """Plot correlation heatmap for top features"""
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        
        # Select top correlated features (based on variance or mutual correlations)
        if n_features < len(numeric_cols):
            # Use feature variance to select most informative features
            variances = X[numeric_cols].var().sort_values(ascending=False)
            selected_cols = variances.head(n_features).index.tolist()
        else:
            selected_cols = numeric_cols
        
        corr_matrix = X[selected_cols].corr(method=method)
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   annot=True, 
                   fmt='.2f', 
                   cmap='RdBu_r', 
                   center=0,
                   square=True, 
                   linewidths=0.5,
                   cbar_kws={'shrink': 0.8})
        
        plt.title(f'Top {len(selected_cols)} Features Correlation Matrix ({method.capitalize()})', 
                 fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        # Show high correlations
        self._show_high_correlations(corr_matrix, threshold)
    
    def _setup_target_controls(self, X: pd.DataFrame, y: pd.Series):
        """Setup interactive controls for target analysis"""
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            display(HTML("<p>No numeric features for target analysis</p>"))
            return
        
        # Create output widget FIRST
        output = widgets.Output()
        
        # Controls
        n_features = widgets.Dropdown(
            options=[5, 10, 15, 20],
            value=15,
            description='Top Features:',
            style={'description_width': 'initial'}
        )
        
        plot_type = widgets.Dropdown(
            options=['Scatter', 'Boxplot', 'Boxen', 'Violin', 'Stripplot'],
            value='Scatter',
            description='Plot Type:',
            style={'description_width': 'initial'}
        )
        
        update_tgt_btn = widgets.Button(description='Update Analysis', button_style='primary')
        
        def on_update_clicked(b):
            with output:
                clear_output(wait=True)
                self._plot_target_relationships(X, y, n_features.value, plot_type.value)
        
        update_tgt_btn.on_click(on_update_clicked)
        
        # Display everything
        display(HTML("<h3>Target Relationship Analysis</h3>"))
        display(widgets.HBox([n_features, plot_type]))
        display(update_tgt_btn)
        display(output)
        
        # Trigger initial plot
        on_update_clicked(None)

    
    def _plot_target_relationships(self, X: pd.DataFrame, y: pd.Series, n_features: int, plot_type: str):
        """Plot relationships between features and target"""
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        
        # Select top correlated features with target
        correlations = X[numeric_cols].corrwith(y).abs().sort_values(ascending=False)
        top_features = correlations.head(n_features).index.tolist()
        
        ncols = 3
        nrows = (len(top_features) + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
        axes_flat = axes.flatten() if nrows * ncols > 1 else [axes]
        
        for i, col in enumerate(top_features):
            if i < len(axes_flat):
                ax = axes_flat[i]
                
                if plot_type == 'Scatter':
                    ax.scatter(X[col], y, alpha=0.5, s=20)
                    ax.set_xlabel(col)
                    ax.set_ylabel('Target')
                elif plot_type in ['Boxplot', 'Boxen', 'Violin', 'Stripplot']:
                    plot_data = pd.DataFrame({col: X[col], 'Target': y})
                    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
                        # Bin target for categorical plots
                        plot_data['Target_Bin'] = pd.cut(y, bins=5)
                        x_var = 'Target_Bin'
                    else:
                        x_var = 'Target'
                    
                    if plot_type == 'Boxplot':
                        sns.boxplot(data=plot_data, x=x_var, y=col, ax=ax)
                    elif plot_type == 'Boxen':
                        sns.boxenplot(data=plot_data, x=x_var, y=col, ax=ax)
                    elif plot_type == 'Violin':
                        sns.violinplot(data=plot_data, x=x_var, y=col, ax=ax)
                    elif plot_type == 'Stripplot':
                        sns.stripplot(data=plot_data, x=x_var, y=col, ax=ax, alpha=0.6, size=3)
                
                corr = correlations[col]
                ax.set_title(f'{col}\n(corr: {corr:.3f})', fontsize=11)
                ax.tick_params(labelsize=9)
        
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
            high_corr_df = pd.DataFrame(high_corr_pairs)
            display(HTML(f"<h4>High Correlations (>{threshold}):</h4>"))
            display(high_corr_df)
        else:
            display(HTML(f"<p>No correlations above {threshold} found</p>"))
