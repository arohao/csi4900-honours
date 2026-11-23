
import sys
import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phishing_pipeline"))

from feature_pipeline import extract_features

app = FastAPI(title="Phishing Detection API")

MODEL_DIR = ROOT / "backend" / "models"

models = {
    "url_only": joblib.load(
        MODEL_DIR / "url_only_random_forest.joblib"
    ),
    "url_html": joblib.load(
        MODEL_DIR / "url_html_random_forest.joblib"
    )
}

schema = json.loads(
    (ROOT / "phishing_pipeline" / "feature_schema.json").read_text()
)

thresholds = json.loads(
    (ROOT / "backend" / "training_thresholds.json").read_text()
)

class ScanRequest(BaseModel):
    mode: Literal["url_only", "url_html"]
    url: str = Field(min_length=1, max_length=4096)
    html: str | None = Field(default=None, max_length=5_000_000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: ScanRequest):

    if not request.url.startswith(("https://", "http://")):
        raise HTTPException(
            status_code=400,
            detail="URL must start with http:// or https://"
        )

    if request.mode == "url_html" and not (request.html and request.html.strip()):
        raise HTTPException(
            status_code=400,
            detail="HTML is required for URL + HTML analysis"
        )

    try:
        features = extract_features(
            url=request.url,
            html=request.html if request.html is not None else "",
            **thresholds
        )

        model = models[request.mode]

        feature_names = list(model.feature_names_in_)

        X = pd.DataFrame(
            [[features[name] for name in feature_names]],
            columns=feature_names
        )

        prediction = int(model.predict(X)[0])
        score = float(
            model.predict_proba(X)[0, list(model.classes_).index(1)]
        )

        return {
            "url": request.url,
            "mode": request.mode,
            "prediction": "phishing" if prediction == 1 else "legitimate",
            "phishing_score": round(score, 4)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        ) from e
