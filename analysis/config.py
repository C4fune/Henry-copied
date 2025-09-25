import os
from typing import Dict, Any

# IMPORTANT: Your API key is INVALID. Please update it with a valid OpenAI API key.
# Get your API key from: https://platform.openai.com/api-keys
# Set it as environment variable: export OPENAI_API_KEY="your-key-here"
# Or replace the string below with your valid API key

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YOUR_API_KEY_HERE')

if OPENAI_API_KEY == 'YOUR_OPENAI_API_KEY_HERE' or 'sk-proj-' not in OPENAI_API_KEY:
    print("="*60)
    print("WARNING: OpenAI API Key not configured!")
    print("Please set your OpenAI API key in one of these ways:")
    print("1. Environment variable: export OPENAI_API_KEY='your-key'")
    print("2. Edit config.py and replace YOUR_OPENAI_API_KEY_HERE")
    print("Get your key at: https://platform.openai.com/api-keys")
    print("="*60)

BIGQUERY_PROJECT_ID = "unique-bonbon-472921-q8"
BIGQUERY_DATASET = "Claims"
PHARMACY_TABLE = "rx_claims"
MEDICAL_TABLE = "medical_claims"

DRUG_CLASSIFICATIONS = {
    'GLP1': ['Mounjaro', 'Ozempic', 'Wegovy', 'Rybelsus', 'Trulicity'],
    'PSORIASIS': ['Tremfya', 'Skyrizi', 'Cosentyx', 'Taltz', 'Rinvoq'],
    'DIABETES': ['Jardiance', 'Farxiga', 'Invokana', 'Steglatro']
}

ANALYSIS_CONFIG = {
    "confidence_threshold": 0.95,
    "min_sample_size": 30,
    "p_value_threshold": 0.05,
    "feature_importance_threshold": 0.05,
    "max_visualization_points": 1000,
    "top_n_default": 10
}

MODEL_CONFIG = {
    "primary_model": "gpt-4o",
    "temperature": 0.1,
    "max_tokens": 4000
}

VIZ_CONFIG = {
    "output_dir": "images",
    "dpi": 150,
    "figure_size": (14, 8),
    "style": "whitegrid",
    "create_dir": True
}