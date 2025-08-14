# Backend README

A Flask API that powers clustering-based persona matching and personalized content generation. The backend serves two workflows:
1) Upload an Excel file for clustering, persona retrieval, and content generation.
2) Generate promotional content directly from campaign inputs without a file upload.

The backend also supports scheduled retraining every quarter at 02:00 to keep clusters and personas fresh.

---

## 1. Tech Stack
- Python 3.10 or newer
- Flask
- scikit-learn, UMAP, HDBSCAN
- Pandas, NumPy, GeoPandas where applicable
- OpenAI Python SDK
- APScheduler or schedule for timed retraining
- gunicorn or waitress for production

---

## 2. Project Structure
```
backend/
  app.py                 # Flask entrypoint and routes
  generate.py            # Core logic for embeddings, clustering, persona lookups, and prompt assembly
  personas/
    cluster_personas.pkl # Persona dictionary keyed by cluster id
  models/
    umap.pkl
    hdbscan.pkl
    encoder.pkl
    scaler.pkl
  uploads/               # Incoming Excel files
  base_data/YYYY-MM-DD/  # Historical training snapshots for retraining
  requirements.txt
  README.md
```
You can rename or relocate folders. If you do, update the environment variables below.

---

## 3. Environment Variables
Create a `.env` file in `backend/` or set these in your host environment:
```
FLASK_ENV=development
FLASK_DEBUG=1
PORT=5000

# External services
OPENAI_API_KEY=sk-...

# Paths
MODEL_DIR=./models
PERSONA_PATH=./personas/cluster_personas.pkl
UPLOAD_DIR=./uploads
BASE_DATA_DIR=./base_data

# CORS
ALLOW_ORIGINS=http://localhost:4200,http://127.0.0.1:4200
```

---

## 4. Installation
```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

If you do not have `requirements.txt` yet, export one:
```bash
pip freeze > requirements.txt
```

---

## 5. Running the Server
### Development
```bash
# With Flask's built-in server
python app.py

# or with Flask CLI
export FLASK_APP=app.py
flask run --port 5000
```

### Production
```bash
# gunicorn example
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

---

## 6. Key Routes

### POST /upload-excel
Uploads an Excel file, applies the preprocessing and clustering pipeline, retrieves the matching persona, and returns generated content.
- Content-Type: multipart/form-data
- Field: file must be .xlsx or .xls (update validation if you want CSV)

Expected columns depend on your trained pipeline. Example:
- date
- location
- loyalty_tier
- Additional campaign fields if required

Response
```json
{
  "status": "ok",
  "cluster_id": 3,
  "persona": { "...": "..." },
  "content": {
    "text": "Generated caption ...",
    "image_prompt": "..."
  }
}
```

### POST /generate-promo
Generates promotional content directly from campaign inputs. No file upload required.
- Body: application/json
- Example body:
```json
{
  "objective": "Drive app installs",
  "industry": "Retail",
  "funnel_stage": "Consideration",
  "past_engagement": "Low",
  "prefer_output": "text"
}
```

### POST /generate-post
Variant of generate-promo used by the prompt page workflow. Expects similar JSON and returns text, image prompt, or both.

---

## 7. Core Logic Summary
generate.py contains reusable functions to:
1. Validate and encode campaign features.
2. Transform features using the fitted encoder and scaler.
3. Project into the reduced space using UMAP.
4. Assign or query cluster labels with HDBSCAN.
5. Retrieve the persona for that cluster from cluster_personas.pkl.
6. Build a high quality prompt that merges persona traits and campaign inputs.
7. Call the OpenAI API to produce text, image prompt, or both.

This design lets the API serve real time results with consistent persona alignment.

---

## 8. Scheduled Retraining
We retrain every quarter at 02:00 Singapore time to avoid peak usage.
Options:
- Use APScheduler inside Flask.
- Or run a separate cron or systemd timer that calls a retrain script.

Pseudocode with APScheduler:
```python
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import pytz

tz = pytz.timezone("Asia/Singapore")
scheduler = BackgroundScheduler(timezone=tz)

def retrain_job():
    # 1) Aggregate all uploads in base_data/
    # 2) Refit encoder, scaler, UMAP, HDBSCAN
    # 3) Regenerate cluster_personas.pkl
    # 4) Replace models atomically
    pass

# Run daily at 02:00 and check if it is first day of quarter
scheduler.add_job(retrain_job, 'cron', hour=2, minute=0)
scheduler.start()
```

Document your chosen approach in code comments.

---

## 9. Testing
- Unit tests for generate.py functions.
- Integration tests for endpoints with Flask's test client.
- Use small sample Excel files under tests/data/.

---

## 10. Deployment Notes
- Set FLASK_ENV=production and disable debug.
- Serve behind Nginx or a cloud load balancer.
- Restrict ALLOW_ORIGINS to your frontend domain.
- Store API keys in a secrets manager when possible.
- Ensure the uploads/ and base_data/ directories are writable.

---

## 11. Troubleshooting
- 415 or 400 on upload: verify Content-Type and file extension checks.
- CORS blocked: confirm ALLOW_ORIGINS and that Flask-CORS is configured.
- Model mismatch: ensure .pkl files match the code version that expects them.
- OpenAI errors: verify OPENAI_API_KEY and request payload.
- Slow first request: warm up models on startup to prime caches.