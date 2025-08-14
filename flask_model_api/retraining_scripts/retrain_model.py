# retrain_model.py

import os
import pandas as pd
import joblib
from datetime import datetime
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import umap
import hdbscan

# Define directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
BASE_DATA_DIR = os.path.join(BASE_DIR, "base_data")
MODEL_DIR = os.path.join(BASE_DIR, "production_models")

# Ensure required directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(BASE_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def run_retraining():
    print("📄 Loading historical + new uploaded data...")
    all_data = []

    # Load base data from versioned folders
    for folder in os.listdir(BASE_DATA_DIR):
        folder_path = os.path.join(BASE_DATA_DIR, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".xlsx"):
                    df = pd.read_excel(os.path.join(folder_path, file))
                    all_data.append(df)
    
    # Load newly uploaded Excel files
    upload_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".xlsx")]
    for file in upload_files:
        df = pd.read_excel(os.path.join(UPLOAD_DIR, file))
        all_data.append(df)

    if not all_data:
        raise ValueError("❌ No data found in uploads or base_data.")

    df_full = pd.concat(all_data, ignore_index=True)
    print("✅ Total merged rows:", df_full.shape[0])
    print("🧾 Sample Customer IDs:", df_full.iloc[:, 0].head())

    # Normalize column names and resolve aliases
    column_aliases = {
        'Customer ID': ['customer id', 'customer_id', 'cust_id', 'id'],
        'Gender': ['gender', 'sex'],
        'Loyalty Tier': ['loyalty tier', 'loyalty_tier', 'tier', 'membership_level'],
        'Date Joined': ['date joined', 'date_joined', 'joined_date', 'ks date', 'join_date'],
        'Location': ['location', 'branch', 'region']
    }

    df_full.columns = [col.strip().lower() for col in df_full.columns]

    for standard_name, aliases in column_aliases.items():
        found = None
        for alias in aliases:
            if alias.lower() in df_full.columns:
                found = alias.lower()
                break
        if found:
            df_full.rename(columns={found: standard_name}, inplace=True)
        else:
            raise ValueError(f"❌ Missing required column: {standard_name}")

    # Data Cleaning & Preprocessing
    df_full.drop_duplicates(inplace=True)
    df_full.dropna(subset=['Customer ID', 'Gender', 'Loyalty Tier', 'Date Joined', 'Location'], inplace=True)
    df_full['Date Joined'] = pd.to_datetime(df_full['Date Joined'], errors='coerce')
    df_full = df_full[df_full['Date Joined'].notna() & (df_full['Date Joined'] <= pd.Timestamp.today())]

    df_full['Gender'] = df_full['Gender'].str.strip().str.title()
    df_full['Location'] = df_full['Location'].str.strip().str.title()
    df_full['Loyalty Tier'] = df_full['Loyalty Tier'].str.strip().str.title()

    df_full = df_full[df_full['Loyalty Tier'].isin(['Silver', 'Gold', 'Platinum'])]
    print("🧹 Cleaned data rows:", df_full.shape[0])

    print("🧠 Engineering features...")
    df_full['Loyalty_Tier_Score'] = df_full['Loyalty Tier'].map({'Silver': 1, 'Gold': 2, 'Platinum': 3})
    df_full['Join_Year'] = pd.to_datetime(df_full['Date Joined']).dt.year
    df_full['Join_Month'] = pd.to_datetime(df_full['Date Joined']).dt.month
    df_full['Join_Quarter'] = pd.to_datetime(df_full['Date Joined']).dt.quarter

    categorical_cols = ['Location', 'Gender', 'Join_Year', 'Join_Month', 'Join_Quarter']
    numerical_cols = ['Loyalty_Tier_Score']
    df_cat = df_full[categorical_cols]
    df_num = df_full[numerical_cols]

    print("🔄 Fitting encoder and scaler...")
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    scaled_cat = encoder.fit_transform(df_cat)

    scaler = StandardScaler()
    scaled_num = scaler.fit_transform(df_num)

    X_combined = np.hstack([scaled_cat, scaled_num])

    print("📦 Loading original UMAP and HDBSCAN parameters...")
    old_umap = joblib.load(os.path.join(MODEL_DIR, "umap_model.pkl"))
    old_clusterer = joblib.load(os.path.join(MODEL_DIR, "HDBSCAN_cluster_model.pkl"))
    umap_params = old_umap.get_params()
    hdbscan_params = old_clusterer.get_params()

    # Refit UMAP with new data
    new_umap = umap.UMAP(**umap_params)
    X_embed = new_umap.fit_transform(X_combined)

    new_clusterer = hdbscan.HDBSCAN(**hdbscan_params)
    clusters = new_clusterer.fit_predict(X_embed)
    df_full['cluster_id'] = clusters

    print("🧠 Generating cluster personas...")
    personas = {}
    for cluster_id in sorted(set(clusters)):
        cluster_data = df_full[df_full['cluster_id'] == cluster_id]
        if cluster_data.empty:
            continue
        persona = {
            'Top_Loyalty_Tier': cluster_data['Loyalty_Tier_Score'].mode().iloc[0],
            'Top_Gender': cluster_data['Gender'].mode().iloc[0],
            'Top_Locations': list(cluster_data['Location'].value_counts().index[:3]),
            'Top_Join_Quarter': cluster_data['Join_Quarter'].mode().iloc[0],
            'Top_Join_Years': list(cluster_data['Join_Year'].value_counts().index[:2])
        }
        personas[cluster_id] = persona

    print(" Saving updated models to model/...")
    joblib.dump(new_umap, os.path.join(MODEL_DIR, "umap_model.pkl"))
    joblib.dump(new_clusterer, os.path.join(MODEL_DIR, "HDBSCAN_cluster_model.pkl"))
    joblib.dump(encoder, os.path.join(MODEL_DIR, "encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(personas, os.path.join(MODEL_DIR, "cluster_personas.pkl"))
    
    # Move processed uploads to dated folder under base_data
    today = datetime.today().strftime("%Y-%m-%d")
    dated_folder = os.path.join(BASE_DATA_DIR, today)
    os.makedirs(dated_folder, exist_ok=True)
    for file in upload_files:
        src = os.path.join(UPLOAD_DIR, file)
        dst = os.path.join(dated_folder, file)
        os.rename(src, dst)

    print(f"✅ Retraining complete. Uploads moved to {dated_folder}")
    print(f"📦 Models saved to {MODEL_DIR}")

#python retrain_model.py
if __name__ == "__main__":
    run_retraining()

