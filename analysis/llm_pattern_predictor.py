"""
LLM-Based Pattern Recognition and Predictive Probability System

This module uses LLM to dynamically recognize prescribing patterns and generate
custom ML models to predict future prescribing behavior without any hardcoded rules.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import openai
import json
from datetime import datetime, timedelta
import os
from config import OPENAI_API_KEY, MODEL_CONFIG

class LLMPatternPredictor:
    """
    Uses LLM to recognize complex prescribing patterns and generate
    predictive models for calculating next-month probabilities.
    """
    
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        if self.api_key and 'sk-' in self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            self.client = None
    
    def predict_next_month_probability(self, df: pd.DataFrame, drug1: str, drug2: str) -> pd.DataFrame:
        """
        Use LLM to analyze prescribing patterns and predict next month's probabilities.
        
        The LLM will:
        1. Analyze the temporal patterns in the data
        2. Identify complex behaviors (seasonality, switching patterns, etc.)
        3. Generate custom code to calculate predictive probabilities
        4. Return probabilities based on the specific patterns found
        """
        
        if not self.client:
            return self._fallback_prediction(df, drug1, drug2)
        
        # Prepare data summary for LLM
        data_summary = self._prepare_data_summary(df, drug1, drug2)
        
        # Generate pattern recognition and prediction code
        prediction_code = self._generate_prediction_code(data_summary, drug1, drug2)
        
        # Execute the LLM-generated code
        results = self._execute_prediction_code(prediction_code, df, drug1, drug2)
        
        return results
    
    def _prepare_data_summary(self, df: pd.DataFrame, drug1: str, drug2: str) -> str:
        """Prepare comprehensive data summary for LLM analysis"""
        
        # Get temporal information
        date_col = self._get_date_column(df)
        npi_col = self._get_npi_column(df)
        drug_col = self._get_drug_column(df)
        
        if date_col:
            df['date'] = pd.to_datetime(df[date_col])
            df['month'] = df['date'].dt.to_period('M')
            
            # Calculate monthly patterns
            monthly_patterns = df.groupby([npi_col, 'month', drug_col]).size().unstack(fill_value=0)
            
            # Get sample prescriber histories
            sample_prescribers = []
            for npi in df[npi_col].unique()[:5]:  # Sample 5 prescribers
                prescriber_data = df[df[npi_col] == npi]
                history = prescriber_data.groupby(['month', drug_col]).size().unstack(fill_value=0)
                
                # Convert Period index to string for JSON serialization
                if not history.empty:
                    history.index = history.index.astype(str)
                    history_dict = {str(k): v for k, v in history.to_dict().items()}
                else:
                    history_dict = {}
                
                sample_prescribers.append({
                    'npi': npi,
                    'history': history_dict,
                    'total_scripts': len(prescriber_data),
                    'drugs_prescribed': prescriber_data[drug_col].unique().tolist()
                })
        else:
            sample_prescribers = []
        
        summary = {
            'total_prescribers': df[npi_col].nunique(),
            'total_prescriptions': len(df),
            'date_range': f"{df[date_col].min()} to {df[date_col].max()}" if date_col else "No dates",
            'drug1_prescribers': df[df[drug_col] == drug1][npi_col].nunique(),
            'drug2_prescribers': df[df[drug_col] == drug2][npi_col].nunique(),
            'sample_prescriber_histories': sample_prescribers,
            'columns': df.columns.tolist()
        }
        
        return json.dumps(summary, default=str)
    
    def _generate_prediction_code(self, data_summary: str, drug1: str, drug2: str) -> str:
        """Use LLM to generate custom prediction code based on patterns"""
        
        prompt = f"""
You are a world-class data scientist specializing in pharmaceutical prescribing analytics.
Analyze the prescribing patterns and generate Python code to predict the probability 
that each doctor will prescribe {drug1} or {drug2} NEXT MONTH.

Data Summary:
{data_summary}

CRITICAL REQUIREMENTS:
1. DO NOT use hardcoded thresholds (like "if 3+ months then probability = 0.75")
2. Recognize complex patterns:
   - Temporal trends (increasing/decreasing prescribing)
   - Seasonality effects
   - Switching behaviors between drugs
   - Prescription volume patterns
   - Recent momentum vs historical baseline
3. Use appropriate statistical/ML methods:
   - Time series analysis (ARIMA, exponential smoothing)
   - Markov chains for state transitions
   - Logistic regression with temporal features
   - Gradient boosting with engineered features
4. Calculate actual predictive probabilities for NEXT MONTH
5. Include confidence intervals based on data quantity/quality

Generate Python code that:
- Takes df, drug1, drug2 as inputs
- Returns a DataFrame with columns:
  - NPI
  - drug1_next_month_probability
  - drug2_next_month_probability
  - drug1_confidence_interval
  - drug2_confidence_interval
  - pattern_description (brief text describing the pattern detected)
  - prediction_method (which model/method was used)

The code should dynamically adapt to the patterns found in the data.
DO NOT use simple rules. Use actual predictive modeling.

Return ONLY executable Python code, no explanations.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_CONFIG["primary_model"],
                messages=[
                    {"role": "system", "content": "You are an expert in predictive analytics and time series forecasting."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=3000
            )
            
            code = response.choices[0].message.content
            # Clean the code
            code = code.replace('```python', '').replace('```', '').strip()
            return code
            
        except Exception as e:
            print(f"LLM generation failed: {e}")
            return self._get_fallback_code()
    
    def _execute_prediction_code(self, code: str, df: pd.DataFrame, drug1: str, drug2: str) -> pd.DataFrame:
        """Execute the LLM-generated prediction code"""
        
        # Prepare namespace with necessary imports
        namespace = {
            'pd': pd,
            'np': np,
            'df': df.copy(),
            'drug1': drug1,
            'drug2': drug2,
            'datetime': datetime,
            'timedelta': timedelta,
        }
        
        # Add statistical libraries
        try:
            import scipy.stats as stats
            namespace['stats'] = stats
        except:
            pass
        
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.preprocessing import StandardScaler
            namespace['LogisticRegression'] = LogisticRegression
            namespace['RandomForestClassifier'] = RandomForestClassifier
            namespace['GradientBoostingClassifier'] = GradientBoostingClassifier
            namespace['StandardScaler'] = StandardScaler
        except:
            pass
        
        try:
            # Execute the generated code
            exec(code, namespace)
            
            # Get the results
            if 'results' in namespace:
                return namespace['results']
            elif 'predictions' in namespace:
                return namespace['predictions']
            else:
                # Look for any DataFrame in namespace
                for key, value in namespace.items():
                    if isinstance(value, pd.DataFrame) and 'probability' in str(value.columns):
                        return value
                        
        except Exception as e:
            print(f"Execution error: {e}")
            print(f"Generated code:\n{code[:500]}...")
        
        return self._fallback_prediction(df, drug1, drug2)
    
    def _get_fallback_code(self) -> str:
        """Fallback prediction code if LLM fails"""
        
        return """
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Prepare temporal features
date_cols = [col for col in df.columns if 'DATE' in col.upper() or 'DD' in col]
npi_cols = [col for col in df.columns if 'NPI' in col.upper()]
drug_cols = [col for col in df.columns if 'DRUG' in col.upper() or 'BRAND' in col.upper()]

date_col = date_cols[0] if date_cols else None
npi_col = npi_cols[0] if npi_cols else 'NPI'
drug_col = drug_cols[0] if drug_cols else 'DRUG'

results = []

if date_col:
    df['date'] = pd.to_datetime(df[date_col])
    df['month'] = df['date'].dt.to_period('M')
    
    # Get unique months
    months = sorted(df['month'].unique())
    recent_months = months[-3:] if len(months) >= 3 else months
    
    for npi in df[npi_col].unique():
        prescriber_data = df[df[npi_col] == npi]
        
        # Calculate trends
        drug1_by_month = prescriber_data[prescriber_data[drug_col] == drug1].groupby('month').size()
        drug2_by_month = prescriber_data[prescriber_data[drug_col] == drug2].groupby('month').size()
        
        # Simple moving average prediction
        drug1_recent = [drug1_by_month.get(m, 0) for m in recent_months]
        drug2_recent = [drug2_by_month.get(m, 0) for m in recent_months]
        
        # Calculate trend
        if len(drug1_recent) >= 2:
            drug1_trend = (drug1_recent[-1] - drug1_recent[0]) / max(len(drug1_recent), 1)
            drug1_pred = max(0, drug1_recent[-1] + drug1_trend)
        else:
            drug1_pred = np.mean(drug1_recent) if drug1_recent else 0
            
        if len(drug2_recent) >= 2:
            drug2_trend = (drug2_recent[-1] - drug2_recent[0]) / max(len(drug2_recent), 1)
            drug2_pred = max(0, drug2_recent[-1] + drug2_trend)
        else:
            drug2_pred = np.mean(drug2_recent) if drug2_recent else 0
        
        # Convert to probability (scripts expected / typical monthly scripts)
        monthly_avg = len(prescriber_data) / max(len(months), 1)
        
        drug1_prob = min(1.0, drug1_pred / max(monthly_avg, 1))
        drug2_prob = min(1.0, drug2_pred / max(monthly_avg, 1))
        
        # Confidence based on data quantity
        confidence = min(0.95, 0.5 + (len(prescriber_data) / 100))
        
        results.append({
            'NPI': npi,
            f'{drug1}_next_month_probability': drug1_prob,
            f'{drug2}_next_month_probability': drug2_prob,
            f'{drug1}_confidence_interval': [max(0, drug1_prob - (1-confidence)), min(1, drug1_prob + (1-confidence))],
            f'{drug2}_confidence_interval': [max(0, drug2_prob - (1-confidence)), min(1, drug2_prob + (1-confidence))],
            'pattern_description': 'Trend-based prediction',
            'prediction_method': 'Moving average with trend'
        })

results = pd.DataFrame(results)
"""
    
    def _fallback_prediction(self, df: pd.DataFrame, drug1: str, drug2: str) -> pd.DataFrame:
        """Fallback prediction if LLM is not available"""
        
        code = self._get_fallback_code()
        namespace = {
            'pd': pd,
            'np': np,
            'df': df.copy(),
            'drug1': drug1,
            'drug2': drug2,
            'datetime': datetime,
            'timedelta': timedelta
        }
        
        exec(code, namespace)
        return namespace.get('results', pd.DataFrame())
    
    def _get_date_column(self, df: pd.DataFrame) -> Optional[str]:
        for col in df.columns:
            if 'DATE' in col.upper() or 'DD' in col:
                return col
        return None
    
    def _get_npi_column(self, df: pd.DataFrame) -> str:
        for col in df.columns:
            if 'NPI' in col.upper():
                return col
        return df.columns[0]
    
    def _get_drug_column(self, df: pd.DataFrame) -> str:
        for col in df.columns:
            if 'DRUG' in col.upper() or 'BRAND' in col.upper():
                return col
        return df.columns[0]
