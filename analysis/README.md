# Analysis Module - Code Structure

## Directory Structure

```
analysis/
│
├── main.py                              # Entry point for pharmaceutical analytics system
│
├── config.py                            # Configuration settings and API keys
│
├── ARCHITECTURE.md                     # Detailed system architecture documentation
│
├── SETUP.md                            # Environment setup instructions
│
├── README.md                           # This file - code structure overview
│
├── explain.txt                         # System explanation and use cases
│
│── Core Processing Modules
├── dynamic_query_processor.py          # Main orchestrator for query routing and processing
├── query_analyzer.py                   # Descriptive analysis code generation via LLM
├── data_loader.py                      # BigQuery data loading with caching
│
│── Predictive & Behavioral Analytics
├── predictive_analyzer.py              # ML model training for predictions
├── behavioral_profiler.py              # Clustering and segmentation analysis
├── advanced_prompts.py                 # Sophisticated prompts for ML code generation
│
│── Probability & Statistical Analysis
├── advanced_probability_analyzer.py    # LLM-based probability calculations with heatmaps
├── llm_pattern_predictor.py           # Dynamic pattern recognition for next-month predictions
├── temporal_probability_calculator.py  # Temporal consistency-based probability scoring
│
│── Visualization
├── visualization.py                    # Multi-panel chart generation
│
│── Utilities
├── bigquery_schema_explorer.py        # Dataset exploration and schema analysis
│
│── Output Directory
└── images/                             # Generated visualizations and reports
    └── [timestamp]_[query]/            # Query-specific output folders
        ├── heatmap_[drug1]_vs_[drug2].png  # 5-graph visualization dashboard
        └── analysis_report.txt              # Detailed quadrant analysis
```

## Module Descriptions

### Core Processing
- **main.py**: Command-line interface for running queries
- **dynamic_query_processor.py**: Routes queries to appropriate analyzers (descriptive, predictive, behavioral, comparative)
- **query_analyzer.py**: Uses GPT-4 to dynamically generate analysis code for each query
- **data_loader.py**: Handles BigQuery connections and provides fallback sample data

### Predictive Analytics
- **predictive_analyzer.py**: Implements ensemble models (XGBoost, LightGBM, Random Forest)
- **behavioral_profiler.py**: K-means, DBSCAN, and hierarchical clustering for prescriber segmentation
- **advanced_prompts.py**: Expert-level prompts for generating statistical and ML code

### Probability Analysis
- **advanced_probability_analyzer.py**: Creates 5-graph dashboards with 16-quadrant heatmaps
- **llm_pattern_predictor.py**: Generates custom predictive models based on prescribing patterns
- **temporal_probability_calculator.py**: Pattern-based probability scoring without hardcoded thresholds

### Visualization
- **visualization.py**: Generates professional pharmaceutical visualizations
- **images/**: Timestamped folders containing PNG visualizations and text reports

## Key Features

1. **Dynamic Code Generation**: LLM generates analysis code on-the-fly for each query
2. **Pattern-Based Scoring**: No hardcoded thresholds, all calculations use continuous functions
3. **Multi-Dataset Integration**: Utilizes 5 BigQuery datasets (rx_claims, medical_claims, providers_bio, provider_payments, us_npi_doctors)
4. **Comprehensive Visualizations**: 5-graph dashboard including heatmaps, scatter plots, bar charts, histograms, and specialty analysis
5. **Robust Fallbacks**: Always provides meaningful results even with limited data
