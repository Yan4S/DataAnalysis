import pandas as pd
import numpy as np
from typing import List, Dict
from IPython.display import display, HTML

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
            reasons = self. (X, col)
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
