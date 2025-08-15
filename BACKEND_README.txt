AI CONTENT PORTAL BACKEND - README

1. PURPOSE
This repository contains a Python Flask API for clustering based persona matching and AI content generation. It serves two workflows that the Angular frontend calls.

Workflow 1  Smart Form
- Frontend sends campaign inputs as JSON
- Backend generates text and optional image prompt
- Endpoint  POST /generate-post or POST /generate-promo

Workflow 2  Upload Dataset
- Frontend uploads an Excel file
- Backend validates, clusters, retrieves personas, and generates content by segment
- Endpoint  POST /upload-excel


2. TECH STACK AND DEPENDENCIES
- Python 3.10 or newer
- Flask and Flask CORS
- Pandas and NumPy
- scikit-learn, UMAP, HDBSCAN
- OpenAI Python SDK
- Joblib or pickle for model files
- Optional APScheduler or schedule for retraining
- gunicorn or waitress for production serving

For the complete list of Python packages and exact versions, refer to requirements.txt in the repository root.

Install them with:
pip install -r requirements.txt


3. REPOSITORY LAYOUT
This reflects your current structure. Names may vary slightly by branch.

backend root
  .venv/                       local virtual environment
  flask_model_api/             optional subfolder if used in your project
  __pycache__/                 Python bytecode cache
  production_models/          trained artifacts and cluster outputs thats being used
  base_data/                   historical training snapshots for retraining
    YYYY-MM-DD folders         store raw or cleaned inputs used for retraining

  model/                       trained artifacts and cluster outputs
    Model_attempts/            notebooks and experiment artifacts(refer to 14 for final ml, 15 for model comparison)
    cluster_profiles.csv       final cluster profile table
    cluster_profiles_*.csv     other versions
    cluster_personas.pkl       dict of personas keyed by cluster id
    umap.pkl                   dimensionality reducer
    hdbscan.pkl                clustering model
    encoder.pkl                categorical encoder if used
    scaler.pkl                 feature scaler if used

  test_model/                  optional test artifacts
  uploads/                     runtime uploads from the frontend

  .env                         environment variables
  app.py                       main Flask entrypoint
  app_v1.py                    older version of the app
  app_v2.py                    older version of the app
  generate.py                  core logic used by app.py
  generate_v1.py               previous version
  generate_v2.py               previous version
  schemas.py                   pydantic models and validation
  test_api.py                  simple smoke tests

  requirements.txt             Python dependencies
  BACKEND_README.txt           this file


4. ENVIRONMENT VARIABLES
Create a .env file in the backend root. Example values below.

FLASK_ENV=development
FLASK_DEBUG=1
PORT=5000

# OpenAI
OPENAI_API_KEY=sk-your-key

# Paths
MODEL_DIR=./model
PERSONA_PATH=./model/cluster_personas.pkl
UPLOAD_DIR=./uploads
BASE_DATA_DIR=./base_data

# CORS
ALLOW_ORIGINS=http://localhost:4200,http://127.0.0.1:4200


5. SETUP
Step 1  create and activate a virtual environment
Windows:
python -m venv .venv
.venv\Scripts\activate

macOS or Linux:
python3 -m venv .venv
source .venv/bin/activate

Step 2  install dependencies
pip install -r requirements.txt

Step 3  create folders if they do not exist
mkdir uploads
mkdir model
mkdir base_data

Step 4  place trained artifacts in model
Required at minimum:
- cluster_personas.pkl
- umap.pkl
- hdbscan.pkl
Add encoder.pkl and scaler.pkl if your pipeline expects them


6. RUNNING
Development:
python app.py

Flask CLI alternative:
set FLASK_APP=app.py    (Windows PowerShell: $env:FLASK_APP="app.py")
flask run --port 5000

Production with gunicorn:
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:5000 app:app


7. API ENDPOINTS

POST /upload-excel
Use for Workflow 2. Accepts an Excel file and returns clusters, personas, and generated content.
Content-Type: multipart/form-data
Field name: file
Accepted: xlsx or xls

Expected columns depend on your pipeline. Common examples:
- date
- location
- loyalty_tier
- other campaign fields as defined in schemas.py

Example curl:
curl -X POST http://localhost:5000/upload-excel ^
  -H "Accept: application/json" ^
  -F "file=@uploads/sample_customer_data.xlsx"

Typical response:
{
  "status": "ok",
  "segments": [
    {
      "cluster_id": 3,
      "persona": { "...": "..." },
      "content": {
        "text": "Generated caption",
        "image_prompt": "..."
      }
    }
  ]
}

POST /generate-post
Use for Workflow 1. Generates content from campaign inputs without a file.
Content-Type: application/json

Example body:
{
  "objective": "Drive app installs",
  "industry": "Retail",
  "funnel_stage": "Consideration",
  "past_engagement": "Low",
  "prefer_output": "text"
}

POST /generate-promo
Alternate route used by the prompt flow. Same idea as generate-post. Your code may route both to a common handler in generate.py.


8. CORE LOGIC OVERVIEW
The generate module performs:
1. Validate and encode inputs using schemas.py
2. Transform features with encoder and scaler if configured
3. Reduce dimensionality with UMAP
4. Assign or query cluster labels with HDBSCAN
5. Map cluster id to persona from cluster_personas.pkl
6. Assemble a prompt that merges persona traits and inputs
7. Call the OpenAI API to produce text, image prompt, or both
8. Return a clean JSON response for the frontend


9. RETRAINING
Goal: keep clusters and personas fresh using data stored in base_data.

Options:
- Run APScheduler inside the Flask app
- Run a separate cron or systemd timer that calls a retraining script

Recommended steps inside your retrain script:
1. Collect all CSV or Excel from base_data
2. Fit encoders and scalers
3. Fit UMAP and HDBSCAN
4. Regenerate personas and cluster profile tables
5. Write new artifacts to a temporary folder
6. Atomically replace files in model after validation
7. Log a short report for audit

Keep retraining code under retraining_script or a similar folder. Document choices in comments.


10. TESTING
- Unit tests for functions in generate.py
- Integration tests for endpoints using Flask test client
- Sample files in a tests or data folder
- A quick smoke test can be run with test_api.py


11. DEPLOYMENT NOTES
- Set FLASK_ENV=production and FLASK_DEBUG=0
- Serve behind Nginx or a managed load balancer
- Restrict ALLOW_ORIGINS to the real frontend domain
- Do not commit .env or model artifacts to public repos
- Ensure uploads and base_data are writable by the service account
- Add startup warming to load models on first request


12. TROUBLESHOOTING
400 or 415 on upload:
- Check multipart form field name "file"
- Ensure xlsx or xls and correct Content-Type

CORS blocked:
- Verify Flask CORS configuration and ALLOW_ORIGINS list

Model mismatch errors:
Pickle model errors or mismatches
- Symptom: server crashes on startup when loading .pkl files. Example:
  TypeError: code() argument 13 must be str, not int
- Cause: the .pkl was created with a different Python or numba stack.
- Fix: rebuild the models in your current environment, then copy the artifacts into the folders below so they contain all 5 files.

Rebuild using the project notebook
1) Open 14_ml_final.ipynb
2) Run all cells to refit UMAP and HDBSCAN and regenerate personas
3) The notebook writes artifacts to your source folder, for example:
   C:\Users\Lenovo\Downloads\Year 3 Major project\ML models\Model_attempts\model
   Files you should have there:
   • cluster_personas.pkl
   • encoder.pkl
   • scaler.pkl
   • umap_model.pkl
   • hdbscan_model.pkl  (your file name may look like HDBSCAN_cluster_*.pkl, that is fine)

Copy into the folders your app reads
Folder A  model\
C:\Users\Lenovo\Downloads\Year 3 Major project\ML models\model
Must contain these 5 files:
• cluster_personas.pkl
• encoder.pkl
• scaler.pkl
• umap_model.pkl
• hdbscan_model.pkl   (or your HDBSCAN_cluster_*.pkl)

Folder B  production_models\
C:\Users\Lenovo\Downloads\Year 3 Major project\ML models\flask_model_api\retraining_scripts\production_models
Must contain these 2 files:
• umap_model.pkl
• hdbscan_model.pkl   (use the same HDBSCAN file, you can rename it to hdbscan_model.pkl)

Note
Do not put model pickles in uploads\

Verify, then restart
dir "C:\Users\Lenovo\Downloads\Year 3 Major project\ML models\model\*.pkl"
dir "C:\Users\Lenovo\Downloads\Year 3 Major project\ML models\flask_model_api\retraining_scripts\production_models\*.pkl"
taskkill /f /im python.exe  >nul 2>&1
python app.py

If the error persists
• You are still loading an old path. Search your code for joblib.load and confirm the filenames.
• Ensure the venv used to run app.py is the same one you used to rebuild the models.

OpenAI errors:
- Verify OPENAI_API_KEY
- Print the prompt payload in debug to see if fields are missing

Slow first request:
- Warm models on startup by calling a dry run function in app.py

Windows script execution denied:
- Run PowerShell as Administrator and set execution policy for the session:
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass


13. SAMPLE .ENV FOR LOCAL DEV
FLASK_ENV=development
FLASK_DEBUG=1
PORT=5000
OPENAI_API_KEY=sk-your-key
MODEL_DIR=./model
PERSONA_PATH=./model/cluster_personas.pkl
UPLOAD_DIR=./uploads
BASE_DATA_DIR=./base_data
ALLOW_ORIGINS=http://localhost:4200
