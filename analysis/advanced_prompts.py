"""
Advanced prompts for predictive analytics and behavioral profiling
"""

PREDICTIVE_ANALYSIS_PROMPT = """
You are a DeepMind-level ML engineer specializing in pharmaceutical predictive analytics.

Query: {query}
Data Available: {data_info}
Current Date: {current_date}
Training Period: Last 3 months unless specified

Generate production-quality Python code for predictive modeling that:

1. FEATURE ENGINEERING:
   - Temporal features: lag variables, rolling averages, trend indicators
   - Prescriber behavior: prescribing velocity, portfolio diversity index
   - Market dynamics: competitive ratios, market share trends
   - Network effects: regional influence scores, peer effects
   - Patient outcome proxies: adherence estimates from refill patterns

2. MODEL SELECTION:
   - For classification: XGBoost + LightGBM + Random Forest ensemble
   - For regression: Gradient boosting + neural network ensemble
   - For time series: Prophet + ARIMA + LSTM
   - Apply appropriate cross-validation (TimeSeriesSplit for temporal data)

3. ADVANCED TECHNIQUES:
   - SMOTE for class imbalance
   - Bayesian hyperparameter optimization
   - Feature selection with SHAP values
   - Uncertainty quantification with conformal prediction
   - Causal inference for treatment effects

4. OUTPUT REQUIREMENTS:
   results = {{
       'predictions': DataFrame with NPI, prediction, confidence,
       'model_performance': {{'auc': 0.85, 'precision@10': 0.92, ...}},
       'feature_importance': DataFrame with SHAP values,
       'prescriber_segments': DataFrame with behavioral clusters,
       'interpretation': "Key insights from the analysis"
   }}

IMPORTANT: Use actual column names from df.columns. Handle missing values appropriately.
Generate executable code that would pass DeepMind code review.
"""

BEHAVIORAL_PROFILING_PROMPT = """
You are a world-class behavioral scientist at DeepMind analyzing prescriber patterns.

Query: {query}
Data Summary: {data_summary}
Focus Drugs: {focus_drugs}

Generate sophisticated behavioral profiling code that:

1. BEHAVIORAL DIMENSIONS:
   - Innovation adoption: early vs late adopter scores
   - Price sensitivity: generic/brand preference patterns
   - Treatment philosophy: aggressive vs conservative
   - Patient complexity: comorbidity handling patterns
   - Peer influence: network effects and regional trends

2. CLUSTERING METHODOLOGY:
   - Multi-algorithm approach: K-means, DBSCAN, Hierarchical
   - Optimal k selection: elbow method + silhouette analysis
   - Stability validation: bootstrap resampling
   - Interpretable archetypes: map clusters to personas

3. COMPETITIVE ANALYSIS (for drug comparisons):
   - Identify switchers, loyalists, and dual prescribers
   - Quantify switching triggers and barriers
   - Predict future switching probability
   - Map competitive dynamics and market share evolution

4. ACTIONABLE OUTPUT:
   profiles = {{
       'clusters': DataFrame with prescriber assignments and archetypes,
       'profiles': Dict with detailed cluster characterization,
       'switching_matrix': DataFrame showing drug preference transitions,
       'targeting_recommendations': List of high-value prescriber segments
   }}

Generate code that creates psychologically valid, actionable behavioral segments.
"""

COMPETITIVE_DRUG_ANALYSIS_PROMPT = """
You are analyzing competitive dynamics between {drug1} and {drug2}.

These drugs are competitors in the {therapeutic_area} market.

Generate code for comprehensive competitive analysis:

1. PRESCRIBER SEGMENTATION:
   - Loyalists: exclusive prescribers of each drug
   - Switchers: changed preference over time
   - Dual prescribers: use both drugs
   - Potential converts: similar profile to switchers

2. PREFERENCE DRIVERS:
   - Patient characteristics associated with each drug
   - Payer/formulary influences
   - Regional variations
   - Prescriber specialty effects

3. PREDICTIVE MODELING:
   - Predict which prescribers will switch
   - Estimate time to switch
   - Quantify switch probability
   - Identify intervention opportunities

4. MARKET DYNAMICS:
   - Share of voice trends
   - Competitive response patterns
   - Market share trajectory projection
   - Win/loss analysis

Output comprehensive insights for strategic decision-making.
"""

TIME_SERIES_FORECAST_PROMPT = """
You are a time series expert building pharmaceutical forecasting models.

Query: {query}
Forecast Horizon: {horizon} months
Historical Data: {data_period}

Generate advanced time series forecasting code:

1. DATA PREPARATION:
   - Handle seasonality and trends
   - Create lag features and rolling statistics
   - External regressors (market events, launches)
   - Handle missing data with forward fill or interpolation

2. MODEL ENSEMBLE:
   - Prophet for trend and seasonality
   - ARIMA/SARIMA for traditional time series
   - LSTM for complex patterns
   - XGBoost for feature-based forecasting

3. VALIDATION:
   - Walk-forward validation
   - Prediction intervals (95% CI)
   - Forecast accuracy metrics (MAPE, RMSE)
   - Scenario analysis

4. INSIGHTS:
   - Trend decomposition
   - Inflection points
   - Growth drivers
   - Risk factors

Generate production-ready forecasting code with interpretable results.
"""

REPORT_GENERATION_PROMPT = """
Generate a comprehensive analytical report for: {query}

Analysis Results: {results}

Create a professional report that includes:

1. EXECUTIVE SUMMARY:
   - Key findings (3-5 bullet points)
   - Strategic implications
   - Recommended actions

2. DETAILED ANALYSIS:
   - Statistical findings with p-values and confidence intervals
   - Behavioral segments and their characteristics
   - Predictive model performance and reliability
   - Feature importance and drivers

3. VISUALIZATIONS:
   - Multi-panel figures with subplots
   - Heatmaps for correlations
   - Time series with confidence bands
   - Cluster visualizations with t-SNE/PCA

4. TECHNICAL APPENDIX:
   - Methodology description
   - Model specifications
   - Validation metrics
   - Data quality assessment

Format as a structured dict with markdown-ready text and visualization specs.
"""

QUERY_CLASSIFICATION_PROMPT = """
Classify this pharmaceutical market research query:

Query: {query}

Determine the query type and requirements:

1. QUERY TYPE:
   - descriptive: current state analysis
   - predictive: future behavior prediction
   - behavioral: prescriber profiling/segmentation
   - competitive: drug vs drug comparison
   - temporal: time series analysis
   - causal: treatment effect estimation

2. DATA REQUIREMENTS:
   - datasets: [rx_claims, medical_claims, providers_bio, ...]
   - time_range: specific period needed
   - minimum_sample_size: for statistical power
   - features_needed: critical columns

3. ANALYTICAL APPROACH:
   - statistical_methods: [t-test, regression, clustering, ...]
   - ml_models: [xgboost, neural_network, ...]
   - visualization_types: [heatmap, time_series, scatter, ...]

4. COMPLEXITY LEVEL:
   - simple: basic statistics
   - moderate: standard ML
   - advanced: ensemble models with feature engineering
   - expert: causal inference or deep learning

Return a JSON with these classifications to route to appropriate analyzer.
"""
