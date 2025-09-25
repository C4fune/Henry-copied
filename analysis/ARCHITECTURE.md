# Advanced Pharmaceutical Analytics Architecture

## System Overview

DeepMind-quality pharmaceutical analytics system combining descriptive analytics with predictive modeling and behavioral profiling. Uses LLM-generated code (GPT-4o) with state-of-the-art ML techniques for comprehensive market intelligence.

## Core Capabilities

### 1. Descriptive Analytics
- **Real-time analysis**: LLM generates custom analysis code for each query
- **Statistical rigor**: Proper p-values, confidence intervals, effect sizes
- **Response time**: 3-5 seconds for standard queries
- **Visualization**: Multi-panel figures, heatmaps, statistical annotations

### 2. Predictive Modeling (NEW)
- **Ensemble ML Models**: XGBoost + LightGBM + Random Forest
- **Time-series forecasting**: Prophet + ARIMA + LSTM for trends
- **Feature engineering**: Behavioral, temporal, market dynamics features
- **Validation**: Time-based cross-validation, SHAP explanations
- **Response time**: 10-15 seconds for predictions
- **Use cases**: Prescriber switching, future prescribing volume, drug adoption

### 3. Behavioral Profiling (NEW)
- **Multi-algorithm clustering**: K-means, DBSCAN, Hierarchical
- **Prescriber archetypes**: Early adopters, conservatives, specialists, generalists
- **Competitive analysis**: Identify switchers vs loyalists
- **Actionable segments**: High-value targeting recommendations
- **Profile characteristics**: Innovation adoption, price sensitivity, treatment philosophy

### 4. Competitive Drug Analysis (NEW)
- **Head-to-head comparisons**: e.g., Rinvoq vs Xeljanz
- **Switching prediction**: Which doctors will switch and when
- **Market dynamics**: Share of voice, win/loss analysis
- **Behavioral drivers**: What distinguishes prescribers of each drug

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Query                           │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    Query Classification                      │
│         (Descriptive / Predictive / Behavioral)             │
└─────────────────────────────────────────────────────────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   Descriptive    │ │   Predictive     │ │   Behavioral     │
│    Analytics     │ │    Analytics     │ │    Profiling     │
└──────────────────┘ └──────────────────┘ └──────────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│               LLM Code Generation (GPT-4o)                  │
│          (Analysis / ML Models / Clustering)                │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    BigQuery Data Loading                    │
│     (rx_claims, medical_claims, providers, payments, npi)   │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                     Code Execution                          │
│            (Dynamic execution of generated code)            │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  Results & Visualizations                   │
│     (Insights, Predictions, Segments, Technical Data)       │
└─────────────────────────────────────────────────────────────┘
```

## Workflow Examples

### Example 1: Predictive Query
**Query**: "Which doctors are likely to prescribe Rinvoq vs Xeljanz next month?"

1. **Classification**: Identified as predictive/competitive query
2. **Data Loading**: Last 3 months of rx_claims for JAK inhibitors
3. **Feature Engineering**: 
   - Prescribing velocity, portfolio diversity
   - Historical Rinvoq/Xeljanz ratios
   - Peer influence scores
4. **Model Training**: XGBoost ensemble with 85% AUC
5. **Output**: Ranked list of prescribers with switch probability

### Example 2: Behavioral Query  
**Query**: "Create behavioral profiles of doctors prescribing diabetes medications"

1. **Classification**: Behavioral profiling query
2. **Data Loading**: All diabetes drug prescriptions
3. **Feature Creation**:
   - Drug diversity index
   - Brand vs generic preference
   - Patient complexity metrics
4. **Clustering**: K-means identifies 5 archetypes
5. **Output**: Detailed profiles with targeting recommendations

## Key Components

### Core Modules
- **dynamic_query_processor.py**: Query routing and orchestration
- **query_analyzer.py**: Descriptive analysis code generation
- **data_loader.py**: BigQuery interface with caching

### Predictive Analytics Modules
- **predictive_analyzer.py**: ML model training and prediction
- **behavioral_profiler.py**: Clustering and segmentation
- **advanced_prompts.py**: Sophisticated ML prompts
- **advanced_probability_analyzer.py**: Statistical probability heatmaps with Bayesian methods

### Support Modules
- **visualization.py**: Multi-panel chart generation
- **config.py**: API keys and model configuration

## Performance Optimizations

### Data Management
- **Smart loading**: Only required datasets and columns
- **Time filtering**: Last 3 months for training (configurable)
- **Caching**: 5-minute cache for repeated queries
- **Sampling**: 100K rows for speed, full data for accuracy

### LLM Optimization
- **Single-pass generation**: Complete code in one LLM call
- **Expert prompts**: DeepMind-level engineering instructions
- **Structured output**: JSON format for reliable parsing

### Model Training
- **Ensemble methods**: Multiple models for robustness
- **Early stopping**: Prevent overfitting
- **Feature selection**: SHAP-based importance ranking
- **Parallel processing**: Multi-core training

## Statistical Rigor

### Descriptive Statistics
- Proper hypothesis testing (t-test, chi-square, ANOVA)
- Effect sizes (Cohen's d, Cramér's V)
- Multiple comparison corrections (Bonferroni)
- Confidence intervals (95% CI)

### Predictive Validation
- Time-based cross-validation (no data leakage)
- Calibration plots for probability estimates
- Precision-recall curves for imbalanced data
- Bootstrap confidence intervals

### Clustering Validation
- Silhouette scores for cluster quality
- Stability analysis via bootstrap
- Gap statistics for optimal k
- Interpretability via feature importance

## Production Features

### Error Handling
- **3-tier fallback system**:
  1. Primary: Full analysis
  2. Secondary: Simplified analysis
  3. Tertiary: Basic statistics
- **Always returns results**: Never fails completely
- **Graceful degradation**: Partial results on timeout

### Monitoring
- Query classification accuracy
- Model performance metrics
- Response time tracking
- Cache hit rates

### Security & Compliance
- API key management via environment variables
- Data access controls
- HIPAA-compliant processing
- Audit logging

## Technology Stack

### Languages & Frameworks
- Python 3.11+
- Pandas, NumPy, SciPy
- Scikit-learn 1.3+
- XGBoost 2.0+
- LightGBM 4.1+

### ML & AI
- OpenAI GPT-4o
- SHAP for explainability
- Optuna for hyperparameter tuning
- Imbalanced-learn for SMOTE

### Data & Visualization
- Google BigQuery
- Matplotlib, Seaborn
- Plotly (optional)

## Future Enhancements

1. **Deep Learning Models**: LSTM/Transformer for sequence modeling
2. **Causal Inference**: Treatment effect estimation
3. **Real-time Streaming**: Live prescription monitoring
4. **Model Registry**: Version control and A/B testing
5. **AutoML Integration**: Automated model selection
6. **Graph Analytics**: Prescriber influence networks