from flask import Flask, request, jsonify
import os
from retrain_model import run_retraining
"""
Retraining API Endpoint (POST /retrain)

This API is not currently used in the main system but is included for future use.
It allows retraining to be triggered via an HTTP request instead of running retrain_model.py manually.

Use case: 
Useful for adding a frontend button, Postman trigger, or automated retraining pipeline later on.
"""
app = Flask(__name__)
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route('/retrain', methods=['POST'])
def retrain():
    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "Empty file list"}), 400

    allowed_extensions = {'.xlsx'}
    for file in files:
        ext = os.path.splitext(file.filename)[1]
        if ext.lower() not in allowed_extensions:
            return jsonify({"error": f"Invalid file type: {file.filename}"}), 400
        file.save(os.path.join(UPLOAD_DIR, file.filename))

    print(" Uploaded files:", [file.filename for file in files])

    try:
        run_retraining()
        return jsonify({"message": "Retraining completed successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
