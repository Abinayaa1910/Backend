import os
import joblib

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "production_models")

def check_file(name):
    path = os.path.join(BASE_DIR, name)
    print(f"\n📦 {name}")
    try:
        obj = joblib.load(path)
        if hasattr(obj, "get_params"):
            print("🔧 Parameters:")
            for k, v in obj.get_params().items():
                print(f" - {k}: {v}")
        elif isinstance(obj, dict):
            print("🧠 Dictionary keys:", list(obj.keys()))
        else:
            print(f"🔍 Object type: {type(obj)}")
    except Exception as e:
        print(f"❌ Failed to load {name}: {e}")

model_files = [
    "umap_model.pkl",
    "HDBSCAN_cluster_model.pkl",
    "encoder.pkl",
    "scaler.pkl",
    "cluster_personas.pkl"
]

for file in model_files:
    check_file(file)
