# Triggering automated model retraining pipeline update
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

def train_model():
    # 1. Load Preprocessed Data
    train_path = os.path.join("heart_disease_preprocessing", "train_preprocessed.csv")
    test_path = os.path.join("heart_disease_preprocessing", "test_preprocessed.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Dataset preprocessed tidak ditemukan. Silakan jalankan preprocessing terlebih dahulu.")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop("target", axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]
    
    # 2. Setup MLflow Tracking
    # Tentukan experiment name jika tidak dijalankan via MLflow CLI
    active_run_id = os.environ.get("MLFLOW_RUN_ID")
    if active_run_id:
        print(f"Menggunakan active run ID: {active_run_id}")
    else:
        mlflow.set_experiment("Heart_Disease_Basic_Classification")
    
    # Aktifkan Autologging
    mlflow.sklearn.autolog()
    
    # 3. Model Training
    print("Memulai pelatihan model dasar dengan MLflow autolog...")
    with mlflow.start_run(run_id=active_run_id, run_name="RandomForest_Basic_Run" if not active_run_id else None) as run:
        # Buat model dasar tanpa hyperparameter tuning
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # 4. Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy on Test Set: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Cari info run id
        run_id = run.info.run_id
        print(f"\nPelatihan selesai. Run ID: {run_id}")

if __name__ == "__main__":
    train_model()
