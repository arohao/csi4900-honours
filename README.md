# Phishing Website Detection Using Machine Learning

## Overview

This project investigates whether combining URL-based features with HTML features improves phishing website detection compared to using URL-based features alone.

The project includes machine learning models, a feature extraction pipeline, experimental evaluations, and a FastAPI backend for making predictions.

The frontend is maintained in a separate repository.

## Technologies

- Python
- scikit-learn
- pandas
- FastAPI
- Uvicorn
- joblib

## Datasets

Two datasets were used:

- **CompPhish:** Used for training and internal validation.
- **PhreshPhish:** Used for independent cross-dataset evaluation.

The models use 70 features in total:
- 47 URL-based features
- 23 HTML-based features

## Models

Two machine learning algorithms were evaluated:

- Logistic Regression
- Random Forest

The experiments compared URL-only, HTML-only, and combined URL + HTML features.

The deployed backend uses two Random Forest models:
- **URL-only:** Analyzes URL features.
- **URL + HTML:** Analyzes both URL and HTML features.

## Final Results

The final evaluation used 3,000 PhreshPhish samples.

| Model | URL-only F1 | URL + HTML F1 |
|---|---|---|
| Logistic Regression | 0.6935 | 0.8052 |
| Random Forest | 0.6984 | 0.8186 |

On a domain-disjoint subset of 2,246 samples:

| Model | URL-only F1 | URL + HTML F1 |
|---|---|---|
| Logistic Regression | 0.6712 | 0.7739 |
| Random Forest | 0.6779 | 0.7817 |

The combined feature models performed better in both evaluations.

## Project Structure

```text
backend/             FastAPI application and saved models
phishing_pipeline/   Feature extraction code and resources
research/            Preprocessing, training, and evaluation scripts
data/                Datasets and configuration files
results/             Predictions and experimental results
docs/                Research documentation
```

## Running the Backend

Create and activate a Python virtual environment, then install the required dependencies.

From the project root:

```bash
python3 -m pip install -r backend/requirements.txt
```

Start the API:

```bash
python3 -m uvicorn backend.main:app --reload
```

Open the API documentation at:

`http://127.0.0.1:8000/docs`

### API Endpoint

**POST `/predict`**

URL-only request:

```json
{
  "mode": "url_only",
  "url": "https://example.com"
}
```

URL + HTML request:

```json
{
  "mode": "url_html",
  "url": "https://example.com",
  "html": "<html><body>Example</body></html>"
}
```

Example response:

```json
{
  "url": "https://example.com",
  "mode": "url_only",
  "prediction": "legitimate",
  "phishing_score": 0.0033
}
```

The phishing score represents the model's estimated score, not a guarantee that a website is safe.

## Documentation

See [Methodology](docs/methodology.md) for details about preprocessing, feature extraction, model evaluation, and experimental results.

## Conclusion

The results suggest that combining URL and HTML features improves phishing detection across datasets. However, performance decreases when models are evaluated on a different dataset, highlighting the challenges of generalization.