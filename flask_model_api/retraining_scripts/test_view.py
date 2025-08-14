import joblib
import os

# Adjust this path based on whether you're inspecting test or live models
umap_model_path = os.path.join("flask_model_api", "retraining_scripts", "test_model", "umap_model.pkl")
# Or if you're now saving to `model/`, use:
# umap_model_path = os.path.join("flask_model_api", "retraining_scripts", "model", "umap_model.pkl")

umap_model = joblib.load(umap_model_path)

print("📉 UMAP Parameters used during retraining:")
for k, v in umap_model.get_params().items():
    print(f"- {k}: {v}")
