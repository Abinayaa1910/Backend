import os
import joblib
from flask import Flask, render_template, request, redirect,jsonify, Response
from dotenv import load_dotenv
from generate import generate_prompt
from generate import generate_prompt_from_editor
from schemas import PromoRequest
from pydantic import ValidationError
from generate import generate_prompt
from flask import Flask
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from generate import generate_prompt_from_persona
import requests   

app = Flask(__name__)
CORS(app)  # This allows all origins
print(app.url_map)

# Load .env from this folder
load_dotenv()

# Get API key (fail-safe)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY is missing from .env")
# Load Gemini key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Load models
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retraining_scripts")
clusterer = joblib.load(os.path.join(BASE_DIR, 'production_models', 'HDBSCAN_cluster_model.pkl'))
encoder = joblib.load(os.path.join(BASE_DIR, 'production_models', 'encoder.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'production_models', 'scaler.pkl'))
umap_model = joblib.load(os.path.join(BASE_DIR, 'production_models', 'umap_model.pkl'))
cluster_personas = joblib.load(os.path.join(BASE_DIR, 'production_models', 'cluster_personas.pkl'))

month_name_to_int = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

@app.route('/generate-promo', methods=['POST'])
def generate_promo():
    try:
        print("🔍 Incoming JSON from frontend:", request.json)

        # Validate and parse input
        parsed = PromoRequest.model_validate(request.json)
        data = parsed.model_dump()
        print("🧪 Number of Variants Requested:", data.get("num_variants"))

        # Convert month name to int and compute quarter
        join_month = data.get("join_month")
        if isinstance(join_month, str):
            join_month_int = month_name_to_int.get(join_month, 1)  # fallback to Jan
            data["join_month"] = join_month_int
            data["join_quarter"] = (join_month_int - 1) // 3 + 1

        platform = data.get("platform", "Instagram")
        post_type = data.get("post_type", "Text")
      

        response = {}

        # STEP 1 — Always generate the prompt
        if post_type == "Image":
            # Don't generate text; just prompt for image
            prompt, _, used_fields, image_urls = generate_prompt(
                user_input=data,
                clusterer=clusterer,
                encoder=encoder,
                scaler=scaler,
                umap_model=umap_model,
                cluster_personas=cluster_personas,
                api_key=OPENAI_API_KEY

            )
        else:
            # For Text or Both, generate both prompt and text
            prompt, result, used_fields , image_urls= generate_prompt(
                user_input=data,
                clusterer=clusterer,
                encoder=encoder,
                scaler=scaler,
                umap_model=umap_model,
                cluster_personas=cluster_personas,
                api_key=OPENAI_API_KEY
            )

        # STEP 2 — Then handle result type
        response["prompt_used"] = prompt
        response["fields_used"] = used_fields

        if post_type == "Text":
            response["generated_result"] = result

        elif post_type == "Image":
            response["generated_result"] = image_urls  # already generated in generate_prompt()

        elif post_type == "Both":
            response["generated_result"] = [
                {
                    "text": text,
                    "image": image
                } for text, image in zip(result, image_urls)
            ]


        else:
            return jsonify({"error": f"Unsupported post_type: {post_type}"}), 400

        return jsonify(response)

    except ValidationError as ve:
        return jsonify({
            'error': "Invalid input",
            'details': ve.errors()
        }), 422

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/upload-excel', methods=['POST'])
def upload_excel():
    try:
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        # Save uploaded file will be used for retraining purposes
        UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        print(f" File saved to {file_path}")

        # Get campaign inputs
        objective = request.form.get('objective', '')
        industry = request.form.get('industry', '')
        funnel_stage = request.form.get('funnelStage', '')
        past_engagement = request.form.get('pastEngagement', '')

        #  Load DataFrame
        df = pd.read_excel(file_path)

        # Filter relevant columns
        keep_cols = ['Date Joined', 'Location', 'Gender', 'Loyalty Tier']
        df = df[keep_cols]
        print("🧹 Selected columns:", list(df.columns))

        # Ordinal encode loyalty
        df['Loyalty_Tier_Score'] = df['Loyalty Tier'].map({
            'Silver': 1, 'Gold': 2, 'Platinum': 3
        })

        #  Feature engineer join date
        df['Join_Year'] = pd.to_datetime(df['Date Joined']).dt.year
        df['Join_Month'] = pd.to_datetime(df['Date Joined']).dt.month
        df['Join_Quarter'] = pd.to_datetime(df['Date Joined']).dt.quarter

        categorical_cols = ['Location', 'Gender', 'Join_Year', 'Join_Month', 'Join_Quarter']
        numerical_cols = ['Loyalty_Tier_Score']

        df_cat = df[categorical_cols]
        df_num = df[numerical_cols]

        #  Encode categoricals
        encoded_cat = encoder.transform(df_cat)

        #  Scale numerical
        scaled_num = scaler.transform(df_num)

        #  Combine all features
        combined = np.hstack([encoded_cat, scaled_num])

        #  Apply UMAP
        df_embed = umap_model.transform(combined)

        #  Cluster
        clusters = clusterer.fit_predict(df_embed)
        df['cluster_id'] = clusters

        #  Group by cluster
        grouped = []
        tier_names = {1: "Silver", 2: "Gold", 3: "Platinum"}

        from collections import Counter
        def most_common(lst):
            lst = list(lst)  # ensures it's a plain list
            return Counter(lst).most_common(1)[0][0] if lst else None

        for cluster_id in sorted(set(clusters)):
            cluster_data = df[df['cluster_id'] == cluster_id]
            cluster_rows = cluster_data.to_dict(orient='records')

            # Dynamically build persona from uploaded data
            persona = {
                "Top_Gender": most_common(cluster_data["Gender"]),
                "Top_Locations": cluster_data["Location"].value_counts().head(3).index.tolist(),
                "Top_Loyalty_Tier": most_common(cluster_data["Loyalty_Tier_Score"]),
                "Top_Join_Quarter": most_common(cluster_data["Join_Quarter"]),
                "Top_Join_Years": cluster_data["Join_Year"].value_counts().head(2).index.tolist(),
                "Top_Join_Months": cluster_data["Join_Month"].value_counts().head(2).index.tolist(),
            }

            tier_score = persona.get('Top_Loyalty_Tier', 1)
            loyalty_label = tier_names.get(tier_score, "General")

            persona_summary = (
                f"{loyalty_label} tier {persona.get('Top_Gender', '').lower()}s "
                f"from {', '.join(persona.get('Top_Locations', [])[:3])} "
                f"who joined in Q{persona.get('Top_Join_Quarter', '?')} "
                f"of {persona.get('Top_Join_Years', ['?'])[0]}"
            )

            try:
                ai_prompt, generated_post = generate_prompt_from_persona(
                    persona_summary=persona_summary,
                    persona=persona,
                    api_key=OPENAI_API_KEY,
                    objective=objective,
                    industry=industry,
                    funnel_stage=funnel_stage,
                    past_engagement=past_engagement
                )

            except Exception as e:
                print(f"❌ AI generation failed for cluster {cluster_id}: {e}")
                generated_post = "⚠️ Failed to generate post."
                ai_prompt = "Prompt unavailable due to error."

            grouped.append({
                "cluster_id": int(cluster_id),
                "persona": persona,
                "persona_summary": persona_summary,
                "default_post": generated_post,
                "prompt_used": ai_prompt,
                "members": cluster_rows,
                "campaign_inputs": {
                    "objective": objective,
                    "industry": industry,
                    "funnel_stage": funnel_stage,
                    "past_engagement": past_engagement
                }
            })

        print("✅ Successfully grouped clusters.")
        return jsonify({"clusters": grouped})

    except Exception as e:
        print(f"❌ Exception during file processing: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate-post', methods=['POST'])
def generate_post():
    try:
        incoming = request.get_json(force=True)

        print("📨 RAW incoming data:", incoming)

        if not isinstance(incoming, dict):
            return jsonify({'error': 'Invalid JSON format: expected a dictionary'}), 400

        # Core fields
        persona_summary = incoming.get("persona_summary", "")
        persona = incoming.get("persona", {})
        cluster_id = incoming.get("cluster_id")
        members = incoming.get("members", [])

        # Campaign inputs (optional)
        campaign_inputs = incoming.get("campaign_inputs", {})
        objective = campaign_inputs.get("objective", "")
        industry = campaign_inputs.get("industry", "")
        funnel_stage = campaign_inputs.get("funnel_stage", "")
        past_engagement = campaign_inputs.get("past_engagement", "")

        if not members or not isinstance(members[0], dict):
            return jsonify({
                "error": "Invalid member format. Each member must be a JSON object/dictionary."
            }), 400

        example_member = members[0]

        #  Call content generation with all inputs
        prompt, result, _, image_urls = generate_prompt(
            user_input=example_member,
            clusterer=clusterer,
            encoder=encoder,
            scaler=scaler,
            umap_model=umap_model,
            cluster_personas=cluster_personas,
            api_key=OPENAI_API_KEY,
            override_persona=persona,
            objective=objective,
            industry=industry,
            funnel_stage=funnel_stage,
            past_engagement=past_engagement
        )

        if not result or (isinstance(result, list) and len(result) == 0):
            result = "⚠️ Post generation failed or returned empty content."

        return jsonify({
            "post": result,
            "prompt_used": prompt,
            "cluster_id": cluster_id,
            "image_url": image_urls
        })

    except Exception as e:
        print(f"❌ Error generating post: {e}")
        return jsonify({'error': str(e)}), 500
    
@app.route("/generate-editor-post", methods=["POST"])
def generate_editor_post():
    data = request.get_json()
    print("📥 Incoming /generate-editor-post payload:")
    print(data)

    persona_summary = data.get("persona_summary", "")
    persona = data.get("persona", {})

    # Campaign inputs
    objective = data.get("objective", "")
    industry = data.get("industry", "")
    funnel_stage = data.get("marketing_funnel_stage", "")
    past_engagement = data.get("past_engagement", "")

    # Content preferences
    platform = data.get("platform", "Instagram")
    post_type = data.get("post_type", "Text")
    tone = data.get("tone", "Professional")
    num_variants = int(data.get("num_variants", 1))

    print("🧠 Parsed Inputs:")
    print(f"- Objective: {objective}")
    print(f"- Industry: {industry}")
    print(f"- Funnel Stage: {funnel_stage}")
    print(f"- Past Engagement: {past_engagement}")
    print(f"- Platform: {platform}")
    print(f"- Post Type: {post_type}")
    print(f"- Tone: {tone}")
    print(f"- Num Variants: {num_variants}")
    print(f"- Persona Summary: {persona_summary[:80]}...")
    print(f"- Persona Keys: {list(persona.keys())}")

    prompt, results, used_fields, image_urls = generate_prompt_from_editor(
        persona_summary,
        persona,
        api_key=os.getenv("OPENAI_API_KEY"),
        objective=objective,
        industry=industry,
        funnel_stage=funnel_stage,
        past_engagement=past_engagement,
        platform=platform,
        post_type=post_type,
        tone=tone,
        num_variants=num_variants
    )

    print("✅ Prompt generated successfully.")
    print("📋 Prompt Preview:")
    print(prompt[:500])  # Print first 500 chars for check
    print("🎯 Used Fields:", used_fields)
    print("📝 Number of Variants:", len(results))
    if image_urls:
        print("🖼️ Image URLs returned:", image_urls)

    return jsonify({
        "prompt": prompt,
        "variants": results,
        "images": image_urls,
        "used_fields": used_fields
    })

@app.route('/api/proxy-download', methods=['POST'])
def proxy_download():
    """
    Expects a JSON payload: { "url": "<full‐SAS‐URL>", "name": "<fileName>.png" }
    Streams the blob back with a Content‐Disposition attachment header.
    """
    data = request.get_json(force=True)
    blob_url = data.get('url')
    filename = data.get('name', 'download.png')

    if not blob_url:
        return jsonify({"error": "Missing 'url' in request body"}), 400

    # Fetch the protected image
    resp = requests.get(blob_url, stream=True)
    if resp.status_code != 200:
        return Response(
            resp.text,
            status=resp.status_code,
            content_type='text/plain'
        )

    # Stream it back as an attachment
    headers = {
        'Content-Disposition': f'attachment; filename="{filename}"'
    }
    return Response(
        resp.raw,
        headers=headers,
        content_type=resp.headers.get('Content-Type', 'application/octet-stream')
    )

#do not remove 
if __name__ == "__main__":
    app.run(debug=True)
