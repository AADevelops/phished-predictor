import os
import sys
from typing import Dict

from fastapi import FastAPI, HTTPException

# make sure the model code is importable
# this points at phished-predictor/model/src
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "model", "src")
    ),
)

from model.src.models.evaluation import evaluate  # noqa: E402

app = FastAPI(
    title="Phishing URL Predictor",
    description="Given feature‐values for a URL, returns phishing probability and top reasons.",
    version="1.0.0",
)


@app.post("/predict")
async def predict(features: Dict[str, float]):
    """
    Expects a JSON object mapping each feature name -> numeric value.
    e.g.
    {
      "NumQuestionMarks": 0.0,
      "DigitLetterRatio": 0.017,
      … 
    }
    """
    try:
        result = evaluate(features)
        return result
    except ValueError as e:
        # bad/missing features
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # missing artifacts or unexpected model error
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # catch‐all
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )