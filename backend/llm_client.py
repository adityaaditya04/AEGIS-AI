"""
Simple LLM client wrapper for the GenAI Firewall.

Behavior:
- If environment variables `LLM_API_URL` and `LLM_API_KEY` are set, forwards prompt
  to that endpoint using a POST request with JSON {"prompt": ...} and expects
  a JSON response containing `text` with the model reply.
- Otherwise, returns a safe mocked response for local development.

This keeps the proxy code decoupled from any single vendor and avoids hardcoding
credentials in the repo.
"""
import os
import logging
import requests
from typing import Dict

logger = logging.getLogger("genai_firewall.llm")

LLM_API_URL = os.environ.get('LLM_API_URL')
LLM_API_KEY = os.environ.get('LLM_API_KEY')


def call_llm(prompt: str, timeout: int = 15) -> Dict[str, str]:
    """Call the configured LLM service and return a dict with `text` reply.

    If no external LLM is configured, return a simple echo/mock response.
    """
    if not prompt:
        return {"text": ""}

    if LLM_API_URL and LLM_API_KEY:
        headers = {
            'Authorization': f'Bearer {LLM_API_KEY}',
            'Content-Type': 'application/json',
        }
        payload = {"prompt": prompt}
        try:
            resp = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            # Try common response shapes
            if isinstance(data, dict):
                if 'text' in data:
                    return {'text': data['text']}
                if 'reply' in data:
                    return {'text': data['reply']}
                # Some LLM APIs return nested choices
                if 'choices' in data and isinstance(data['choices'], list) and len(data['choices']) > 0:
                    choice = data['choices'][0]
                    if isinstance(choice, dict) and 'text' in choice:
                        return {'text': choice['text']}
            # Fallback: return stringified JSON
            return {'text': str(data)}
        except Exception as e:
            logger.exception(f"LLM call failed: {e}")
            return {'text': f"[llm_error] {e}"}

    # Mock behavior for local dev
    logger.info("LLM not configured - returning mock response")
    safe_reply = f"(mock) The model would reply to: {prompt[:200]}"
    return {'text': safe_reply}
