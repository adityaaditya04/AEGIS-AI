import os
from backend.classifier import PromptClassifier


def test_classifier_loads_and_predicts():
    # Use model path if available; otherwise the classifier will be unavailable
    clf = PromptClassifier()
    # Should not raise when calling predict on a simple text
    label, score = clf.predict('Hello, how are you?')
    assert isinstance(label, int)
    assert isinstance(score, float)
