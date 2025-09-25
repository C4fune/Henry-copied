import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import json
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_curve, roc_auc_score,
    f1_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error
)
import xgboost as xgb
import lightgbm as lgb
from config import OPENAI_API_KEY, MODEL_CONFIG
import openai


class PredictiveAnalyzer:
    """DeepMind-quality predictive analytics for pharmaceutical data"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.models = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
    def generate_prediction_code(self, query: str, data_info: Dict[str, Any]) -> str:
        """Generate sophisticated ML prediction code using LLM"""
        
        # Extract key information about the data
        columns = data_info.get('columns', [])
        records = data_info.get('records', 0)
        drugs = data_info.get('drugs', {})
        
        prompt = f"""
        You are a Senior ML Engineer at DeepMind's Healthcare division, specializing in pharmaceutical predictive analytics.
        
        TASK: Generate WORKING predictive model code for this query:
        "{query}"
        
        DATA AVAILABLE:
        - Records: {records:,}
        - Columns: {columns[:20]}  # Show first 20 columns
        - Top drugs: {list(drugs.keys())[:10]}
        - Current date: {datetime.now().strftime('%Y-%m-%d')}
        
        MANDATORY REQUIREMENTS:
        1. Code MUST execute without errors
        2. ALWAYS provide predictions, even with limited data
        3. Use actual column names from the data
        4. Handle missing values gracefully
        5. Return meaningful insights, not generic messages
        
        FEATURE ENGINEERING (use what's available):
        - Prescriber metrics: script counts, drug diversity, avg values
        - Temporal features: month, quarter, trends IF date columns exist
        - Drug-specific: market share, switching patterns
        - Geographic: state-level aggregates if available
        - Specialty effects: segment-specific patterns
        
        MODEL IMPLEMENTATION:
        ```python
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
        
        # Feature engineering
        features = pd.DataFrame()
        # [Build features from actual columns]
        
        # Model training
        X_train, X_test, y_train, y_test = train_test_split(...)
        
        # Ensemble prediction
        models = {{}}
        # Train multiple models
        
        # Generate results
        predictions = pd.DataFrame({{
            'prescriber_id': [...],
            'prediction_probability': [...],
            'predicted_class': [...],
            'confidence': [...]
        }})
        
        model_performance = {{
            'accuracy': accuracy_score(...),
            'auc_roc': roc_auc_score(...),
            'f1_score': f1_score(...)
        }}
        
        feature_importance = pd.DataFrame({{
            'feature': [...],
            'importance': [...],
            'description': [...]
        }})
        
        behavioral_segments = pd.DataFrame({{
            'segment': [...],
            'characteristics': [...],
            'size': [...],
            'targeting_priority': [...]
        }})
        ```
        
        CRITICAL: 
        - For Rinvoq vs Xeljanz: Create binary classification (prefers Rinvoq = 1)
        - For switching prediction: Use historical patterns to predict future
        - For market evolution: Time-series forecasting
        
        OUTPUT MUST INCLUDE:
        - predictions: Specific prescriber predictions
        - model_performance: Real metrics (not placeholders)
        - feature_importance: Actual feature contributions
        - behavioral_segments: Meaningful prescriber groups
        
        Return ONLY executable Python code. NO explanations.
        """
        
        response = self.client.chat.completions.create(
            model=MODEL_CONFIG["primary_model"],
            messages=[
                {"role": "system", "content": "You are a DeepMind ML engineer specializing in pharmaceutical predictive analytics. Generate production-quality ML code with state-of-the-art techniques."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=3000
        )
        
        return response.choices[0].message.content.replace('```python', '').replace('```', '').strip()
    
    def train_prescriber_prediction_model(self, df: pd.DataFrame, target_drug: str, 
                                         competitor_drug: Optional[str] = None,
                                         prediction_window: int = 30) -> Dict[str, Any]:
        """Train model to predict prescriber behavior"""
        
        # Generate sophisticated features
        features_df = self._engineer_prescriber_features(df, target_drug, competitor_drug)
        
        # Prepare target variable
        if competitor_drug:
            # Binary classification: predict preference
            y = self._create_preference_target(df, target_drug, competitor_drug)
            model_type = 'classification'
        else:
            # Regression: predict future prescribing volume
            y = self._create_volume_target(df, target_drug, prediction_window)
            model_type = 'regression'
        
        # Time-based split
        train_size = int(len(features_df) * 0.7)
        X_train = features_df[:train_size]
        X_test = features_df[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        # Train ensemble model
        if model_type == 'classification':
            models = self._train_classification_ensemble(X_train, y_train, X_test, y_test)
            predictions, performance = self._evaluate_classification(models, X_test, y_test)
        else:
            models = self._train_regression_ensemble(X_train, y_train, X_test, y_test)
            predictions, performance = self._evaluate_regression(models, X_test, y_test)
        
        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(models, X_train.columns)
        
        return {
            'predictions': predictions,
            'performance': performance,
            'feature_importance': feature_importance,
            'model_type': model_type,
            'models': models
        }
    
    def _engineer_prescriber_features(self, df: pd.DataFrame, 
                                     target_drug: str,
                                     competitor_drug: Optional[str]) -> pd.DataFrame:
        """Create sophisticated prescriber behavioral features"""
        
        features = pd.DataFrame()
        
        # Basic prescriber metrics
        prescriber_stats = df.groupby('PRESCRIBER_NPI_NBR').agg({
            'TOTAL_PAID_AMT': ['mean', 'std', 'sum'],
            'DISPENSED_QUANTITY_VAL': ['mean', 'std', 'sum'],
            'DAYS_SUPPLY_VAL': ['mean', 'std'],
            'claim_count': 'sum'
        }).reset_index()
        prescriber_stats.columns = ['_'.join(col).strip() for col in prescriber_stats.columns.values]
        features = prescriber_stats
        
        # Drug portfolio diversity (Herfindahl index)
        drug_diversity = df.groupby('PRESCRIBER_NPI_NBR')['NDC_PREFERRED_BRAND_NM'].apply(
            lambda x: 1 / (x.value_counts(normalize=True) ** 2).sum()
        ).reset_index(name='drug_diversity_index')
        features = features.merge(drug_diversity, left_on='PRESCRIBER_NPI_NBR_', right_on='PRESCRIBER_NPI_NBR')
        
        # Temporal patterns
        if 'RX_ANCHOR_DD' in df.columns:
            df['month'] = pd.to_datetime(df['RX_ANCHOR_DD']).dt.month
            df['quarter'] = pd.to_datetime(df['RX_ANCHOR_DD']).dt.quarter
            
            # Prescribing velocity (scripts per month)
            velocity = df.groupby(['PRESCRIBER_NPI_NBR', 'month']).size().reset_index(name='monthly_scripts')
            velocity_stats = velocity.groupby('PRESCRIBER_NPI_NBR')['monthly_scripts'].agg(['mean', 'std']).reset_index()
            velocity_stats.columns = ['PRESCRIBER_NPI_NBR', 'prescribing_velocity_mean', 'prescribing_velocity_std']
            features = features.merge(velocity_stats, left_on='PRESCRIBER_NPI_NBR_', right_on='PRESCRIBER_NPI_NBR', suffixes=('', '_drop'))
            features = features.drop(columns=[col for col in features.columns if col.endswith('_drop')])
        
        # Target drug affinity
        target_stats = df[df['NDC_PREFERRED_BRAND_NM'] == target_drug].groupby('PRESCRIBER_NPI_NBR').agg({
            'TOTAL_PAID_AMT': 'sum',
            'claim_count': 'sum'
        }).reset_index()
        target_stats.columns = ['PRESCRIBER_NPI_NBR', f'{target_drug}_total_paid', f'{target_drug}_scripts']
        features = features.merge(target_stats, left_on='PRESCRIBER_NPI_NBR_', right_on='PRESCRIBER_NPI_NBR', how='left', suffixes=('', '_drop'))
        features = features.drop(columns=[col for col in features.columns if col.endswith('_drop')])
        features.fillna(0, inplace=True)
        
        # Competitor drug stats if applicable
        if competitor_drug:
            comp_stats = df[df['NDC_PREFERRED_BRAND_NM'] == competitor_drug].groupby('PRESCRIBER_NPI_NBR').agg({
                'TOTAL_PAID_AMT': 'sum',
                'claim_count': 'sum'
            }).reset_index()
            comp_stats.columns = ['PRESCRIBER_NPI_NBR', f'{competitor_drug}_total_paid', f'{competitor_drug}_scripts']
            features = features.merge(comp_stats, left_on='PRESCRIBER_NPI_NBR_', right_on='PRESCRIBER_NPI_NBR', how='left', suffixes=('', '_drop'))
            features = features.drop(columns=[col for col in features.columns if col.endswith('_drop')])
            features.fillna(0, inplace=True)
            
            # Competitive ratio
            features['competitive_ratio'] = features[f'{target_drug}_scripts'] / (features[f'{competitor_drug}_scripts'] + 1)
        
        # Specialty encoding
        if 'PRESCRIBER_NPI_HCP_SEGMENT_DESC' in df.columns:
            specialty_dummies = pd.get_dummies(df.groupby('PRESCRIBER_NPI_NBR')['PRESCRIBER_NPI_HCP_SEGMENT_DESC'].first(), prefix='specialty')
            features = pd.concat([features, specialty_dummies], axis=1)
        
        # Regional features
        if 'PRESCRIBER_NPI_STATE_CD' in df.columns:
            state_prescribing = df.groupby('PRESCRIBER_NPI_STATE_CD')['claim_count'].sum().reset_index(name='state_total_scripts')
            prescriber_states = df.groupby('PRESCRIBER_NPI_NBR')['PRESCRIBER_NPI_STATE_CD'].first().reset_index()
            prescriber_states = prescriber_states.merge(state_prescribing, on='PRESCRIBER_NPI_STATE_CD')
            features = features.merge(prescriber_states[['PRESCRIBER_NPI_NBR', 'state_total_scripts']], 
                                    left_on='PRESCRIBER_NPI_NBR_', right_on='PRESCRIBER_NPI_NBR', how='left', suffixes=('', '_drop'))
            features = features.drop(columns=[col for col in features.columns if col.endswith('_drop')])
        
        # Payer mix features
        if 'PAYER_PLAN_CHANNEL_NM' in df.columns:
            payer_mix = df.pivot_table(
                index='PRESCRIBER_NPI_NBR',
                columns='PAYER_PLAN_CHANNEL_NM',
                values='claim_count',
                aggfunc='sum',
                fill_value=0
            ).reset_index()
            payer_mix.columns = [f'payer_{col}' if col != 'PRESCRIBER_NPI_NBR' else col for col in payer_mix.columns]
            features = features.merge(payer_mix, left_on='PRESCRIBER_NPI_NBR_', right_on='PRESCRIBER_NPI_NBR', how='left', suffixes=('', '_drop'))
            features = features.drop(columns=[col for col in features.columns if col.endswith('_drop')])
        
        # Drop redundant NPI columns
        npi_cols = [col for col in features.columns if 'PRESCRIBER_NPI' in col and col != 'PRESCRIBER_NPI_NBR_']
        features = features.drop(columns=npi_cols)
        
        # Ensure all columns are numeric
        for col in features.columns:
            if features[col].dtype == 'object':
                features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
        
        return features
    
    def _create_preference_target(self, df: pd.DataFrame, drug1: str, drug2: str) -> np.ndarray:
        """Create binary target for drug preference prediction"""
        
        # Get all unique prescribers
        all_prescribers = df['PRESCRIBER_NPI_NBR'].unique()
        
        drug1_scripts = df[df['NDC_PREFERRED_BRAND_NM'] == drug1].groupby('PRESCRIBER_NPI_NBR')['claim_count'].sum()
        drug2_scripts = df[df['NDC_PREFERRED_BRAND_NM'] == drug2].groupby('PRESCRIBER_NPI_NBR')['claim_count'].sum()
        
        # Reindex to ensure same prescribers
        drug1_scripts = drug1_scripts.reindex(all_prescribers, fill_value=0)
        drug2_scripts = drug2_scripts.reindex(all_prescribers, fill_value=0)
        
        preference = (drug1_scripts > drug2_scripts).astype(int)
        return preference.values
    
    def _create_volume_target(self, df: pd.DataFrame, drug: str, window: int) -> np.ndarray:
        """Create regression target for future prescribing volume"""
        
        # Get all unique prescribers
        all_prescribers = df['PRESCRIBER_NPI_NBR'].unique()
        
        # This would ideally use time-shifted data
        future_scripts = df[df['NDC_PREFERRED_BRAND_NM'] == drug].groupby('PRESCRIBER_NPI_NBR')['claim_count'].sum()
        future_scripts = future_scripts.reindex(all_prescribers, fill_value=0)
        
        return future_scripts.values
    
    def _train_classification_ensemble(self, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
        """Train ensemble of classification models"""
        
        models = {}
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        models['xgboost'] = xgb_model
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )
        lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        models['lightgbm'] = lgb_model
        
        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        models['random_forest'] = rf_model
        
        # Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        models['gradient_boosting'] = gb_model
        
        return models
    
    def _train_regression_ensemble(self, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
        """Train ensemble of regression models"""
        
        models = {}
        
        # XGBoost Regressor
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        models['xgboost'] = xgb_model
        
        # LightGBM Regressor
        lgb_model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )
        lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        models['lightgbm'] = lgb_model
        
        # Random Forest Regressor
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        models['random_forest'] = rf_model
        
        return models
    
    def _evaluate_classification(self, models: Dict, X_test, y_test) -> Tuple[pd.DataFrame, Dict]:
        """Evaluate classification models"""
        
        predictions = pd.DataFrame()
        performance = {}
        
        # Ensemble predictions
        ensemble_probs = []
        for name, model in models.items():
            probs = model.predict_proba(X_test)[:, 1]
            ensemble_probs.append(probs)
            
            # Individual model performance
            y_pred = model.predict(X_test)
            performance[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'auc_roc': roc_auc_score(y_test, probs)
            }
        
        # Ensemble average
        ensemble_prob = np.mean(ensemble_probs, axis=0)
        ensemble_pred = (ensemble_prob > 0.5).astype(int)
        
        performance['ensemble'] = {
            'accuracy': accuracy_score(y_test, ensemble_pred),
            'f1_score': f1_score(y_test, ensemble_pred),
            'auc_roc': roc_auc_score(y_test, ensemble_prob)
        }
        
        predictions['probability'] = ensemble_prob
        predictions['prediction'] = ensemble_pred
        predictions['confidence'] = np.abs(ensemble_prob - 0.5) * 2
        
        return predictions, performance
    
    def _evaluate_regression(self, models: Dict, X_test, y_test) -> Tuple[pd.DataFrame, Dict]:
        """Evaluate regression models"""
        
        predictions = pd.DataFrame()
        performance = {}
        
        # Ensemble predictions
        ensemble_preds = []
        for name, model in models.items():
            preds = model.predict(X_test)
            ensemble_preds.append(preds)
            
            # Individual model performance
            performance[name] = {
                'mae': mean_absolute_error(y_test, preds),
                'mse': mean_squared_error(y_test, preds),
                'rmse': np.sqrt(mean_squared_error(y_test, preds))
            }
        
        # Ensemble average
        ensemble_pred = np.mean(ensemble_preds, axis=0)
        
        performance['ensemble'] = {
            'mae': mean_absolute_error(y_test, ensemble_pred),
            'mse': mean_squared_error(y_test, ensemble_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred))
        }
        
        predictions['prediction'] = ensemble_pred
        predictions['std'] = np.std(ensemble_preds, axis=0)
        predictions['confidence'] = 1 / (1 + predictions['std'])
        
        return predictions, performance
    
    def _calculate_feature_importance(self, models: Dict, feature_names: List[str]) -> pd.DataFrame:
        """Calculate aggregated feature importance across ensemble"""
        
        importance_dict = {}
        
        for name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                for i, feat_name in enumerate(feature_names):
                    if feat_name not in importance_dict:
                        importance_dict[feat_name] = []
                    importance_dict[feat_name].append(importance[i])
        
        # Average importance across models
        feature_importance = pd.DataFrame([
            {'feature': feat, 'importance': np.mean(scores), 'std': np.std(scores)}
            for feat, scores in importance_dict.items()
        ])
        
        return feature_importance.sort_values('importance', ascending=False)
