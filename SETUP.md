# Setup Instructions

## Environment Variables

Set the following environment variables before running the system:

### OpenAI API Key (Required)
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

### Google Cloud Authentication (Optional)
For BigQuery access, authenticate using one of these methods:

```bash
# Option 1: Use gcloud CLI
gcloud auth application-default login

# Option 2: Use service account
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the system:
```bash
cd analysis
python main.py
```

## Note
If BigQuery authentication is not configured, the system will use sample data for demonstration purposes.
