"""
FastAPI app for the GenAI Firewall bouncer (basic version).

Endpoints:
- GET /health            : returns simple health JSON
- POST /classify         : accepts {"text": "..."} and returns classification result

Run locally:
  uvicorn backend.main:app --reload --port 8000

The app loads the model via `backend.classifier.classifier_singleton`.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging

from .classifier import classifier_singleton
from .llm_client import call_llm

logger = logging.getLogger("genai_firewall.api")

app = FastAPI(title="GenAI Firewall - Bouncer API", version="0.1")

# Allow local frontend during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PromptIn(BaseModel):
    text: str


class ClassifyOut(BaseModel):
    label: str  # 'malicious' or 'safe'
    score: float
    blocked: bool


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": classifier_singleton.is_available()}


@app.post("/classify", response_model=ClassifyOut)
def classify(payload: PromptIn):
    text = payload.text
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided")

    label_num, score = classifier_singleton.predict(text)
    label = "malicious" if label_num == 1 else "safe"
    blocked = label_num == 1

    # Additional security logging could be placed here (audit, reason, etc.)
    logger.info(f"Classify request: label={label} score={score:.4f}")

    return ClassifyOut(label=label, score=round(score, 4), blocked=blocked)


class ProxyIn(BaseModel):
    text: str


class ProxyOut(BaseModel):
    text: str


@app.post("/proxy", response_model=ProxyOut)
def proxy(payload: ProxyIn):
    """Proxy a safe prompt to the configured LLM service.

    This endpoint checks the prompt with the classifier and only forwards
    if it is considered safe. If the model is not loaded, it returns an
    error to avoid sending unfiltered prompts to the LLM.
    """
    text = payload.text
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided")

    if not classifier_singleton.is_available():
        # Fail closed: do not forward prompts if classifier unavailable
        raise HTTPException(status_code=503, detail="Classifier unavailable; cannot proxy prompts")

    label_num, score = classifier_singleton.predict(text)
    if label_num == 1:
        raise HTTPException(status_code=403, detail="Prompt classified as malicious and blocked")

    # Forward to LLM client wrapper
    reply = call_llm(text)
    return ProxyOut(text=reply.get('text', ''))


class ThresholdIn(BaseModel):
    threshold: float


@app.post("/admin/threshold")
def set_threshold(payload: ThresholdIn):
    """Dev endpoint to set the classifier threshold at runtime.

    This is intentionally simple for local development. In production you
    should protect this endpoint with authentication and consider safer
    configuration methods.
    """
    val = float(payload.threshold)
    if val < 0.0 or val > 1.0:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")
    classifier_singleton.threshold = val
    return {"threshold": classifier_singleton.threshold}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
