# Aegis AI

Repository: https://github.com/adityaaditya04/AEGIS-AI

GenAI Firewall (Aegis AI) — a prompt-injection detector and proxy ("bouncer") that sits between a web chat UI and an LLM API.

Features
- TF-IDF + Logistic Regression baseline classifier to detect malicious prompt injections
- FastAPI backend that classifies prompts and proxies safe prompts to an LLM
- Rule-based stop-gap for reducing false positives, runtime threshold tuning, and calibration utilities
- Simple vanilla JS frontend chat UI for local testing

Quick start
1. Create and activate a Python venv (Windows example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
# or
pip install pandas scikit-learn joblib fastapi uvicorn requests pytest
```

3. Train the initial model and save it (uses `data/train.csv` by default):

```powershell
python src\train_model.py --data-path data\train.csv --output models\initial_model.pkl
```

4. (Optional) Calibrate the model:

```powershell
python src\calibrate_model.py --model models\initial_model.pkl --data data\train.csv --method sigmoid
```

5. Start the FastAPI bouncer:

```powershell
uvicorn backend.main:app --reload --port 8000
```

6. Serve the frontend (Flask dev server):

```powershell
python web_app\app.py
# open http://127.0.0.1:3000
```

CI
 - ![CI](https://github.com/adityaaditya04/AEGIS-AI/actions/workflows/ci.yml/badge.svg)

Admin endpoint protection
 - You can protect the runtime admin endpoint `POST /admin/threshold` by setting an environment variable `ADMIN_TOKEN` before starting the FastAPI server. When `ADMIN_TOKEN` is set, include the header `X-Admin-Token: <value>` in requests to `/admin/threshold`.

Development notes
- Adjust classifier threshold at runtime (dev): `POST /admin/threshold` with JSON `{ "threshold": 0.95 }`.
- Diagnostics: `python src/diagnose_model.py` shows false positives and feature contributions.
- Tests: `pytest -q`

Repository: https://github.com/adityaaditya04/AEGIS-AI

License
- Add an appropriate LICENSE file if you plan to release this publicly.

<!-- ci trigger: whitespace -->
