"""
Model loader and inference helper for the GenAI Firewall bouncer.

Responsibilities:
- Load a saved scikit-learn pipeline (joblib .pkl)
- Provide convenient methods for scoring a single prompt or batch
- Apply a configurable threshold to decide whether to block

This module is intentionally small and dependency-light so it can be
imported inside a FastAPI app and reused in worker threads.
"""

import os
import logging
from typing import List, Tuple, Optional

import joblib

logger = logging.getLogger("genai_firewall.classifier")


class PromptClassifier:
    def __init__(self, model_path: Optional[str] = None, threshold: float = 0.5):
        """Initialize and (try to) load the model pipeline.

        Args:
            model_path: Path to joblib .pkl pipeline. If None, looks for
                        ../models/initial_model.pkl relative to this file.
            threshold: Probability threshold above which a prompt is marked malicious.
        """
        if model_path is None:
            here = os.path.dirname(__file__)
            # Prefer a calibrated model if available
            calibrated = os.path.abspath(os.path.join(here, '..', 'models', 'calibrated_model.pkl'))
            default = os.path.abspath(os.path.join(here, '..', 'models', 'initial_model.pkl'))
            model_path = calibrated if os.path.exists(calibrated) else default

        self.model_path = model_path
        self.threshold = float(threshold)
        self.pipeline = None

        self._load()

    def _load(self) -> None:
        """Load the joblib pipeline from disk. If not found, pipeline remains None."""
        if not os.path.exists(self.model_path):
            logger.warning(f"Model file not found at {self.model_path}. Classifier will be unavailable.")
            return
        try:
            self.pipeline = joblib.load(self.model_path)
            logger.info(f"Loaded model pipeline from {self.model_path}")
        except Exception as e:
            logger.exception(f"Failed to load model from {self.model_path}: {e}")
            self.pipeline = None

    def is_available(self) -> bool:
        return self.pipeline is not None

    def score(self, texts: List[str]) -> List[float]:
        """Return probability that each text is malicious (float in [0,1]).

        If the pipeline is not available, returns 0.0 for each input and logs a warning.
        """
        if not self.is_available():
            logger.warning("Called score() but pipeline is not loaded - returning zeros")
            return [0.0 for _ in texts]

        # Many scikit-learn classifiers expose predict_proba
        try:
            probs = self.pipeline.predict_proba(texts)
            # Default to second column if available
            if probs.shape[1] == 2:
                malicious_scores = [float(p[1]) for p in probs]
            else:
                malicious_index = 1
                malicious_scores = [float(p[malicious_index]) if len(p) > malicious_index else 0.0 for p in probs]

            # Apply lightweight rule-based adjustments to reduce false positives
            adjusted = [self._apply_rule_adjustments(text, score) for text, score in zip(texts, malicious_scores)]
            return adjusted
        except Exception as e:
            logger.exception(f"Error during predict_proba: {e}")
            return [0.0 for _ in texts]

    def _apply_rule_adjustments(self, text: str, score: float) -> float:
        """Apply simple heuristics to reduce false positives for common news/article patterns.

        Heuristics implemented:
        - If text contains common news source markers like '(Reuters)', 'AP', 'Reuters', 'By ' at start,
          decrease score by a fixed amount (news_boost_reduce).
        - If text is long (>=300 chars) and looks like an article (contains multiple sentences),
          slightly reduce the score because news articles often contain topical words.

        These are conservative and tunable; they are meant to be a stop-gap while
        we retrain or calibrate the model.
        """
        t = text or ""
        s = float(score)
        lowered = s

        # Rule: obvious news dateline or source mentions
        news_markers = ['(Reuters)', 'Reuters', '\nReuters', 'AP -', 'Associated Press', 'By ', 'Source:']
        if any(marker in t for marker in news_markers):
            lowered = max(0.0, lowered - 0.30)

        # Rule: long article-like text — reduce modestly
        if len(t) >= 300:
            lowered = max(0.0, lowered - 0.10)

        # Clamp
        return float(min(max(lowered, 0.0), 1.0))

    def predict(self, text: str) -> Tuple[int, float]:
        """Return (label, score) where label=1 indicates malicious.

        Uses the configured threshold on the malicious probability.
        """
        score = self.score([text])[0]
        label = 1 if score >= self.threshold else 0
        return label, score


# Convenience singleton instance used by the FastAPI app. The app may
# import this module and use `classifier_singleton` directly.
classifier_singleton = PromptClassifier()
