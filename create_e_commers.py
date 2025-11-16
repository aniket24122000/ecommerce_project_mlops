import os

folders = [
    "data/raw",
    "data/ingested",
    "data/processed",
    "data/features",
    "data/models",
    "notebooks",
    "src/config",
    "app"
]

files = [
    "src/data_ingestion.py",
    "src/preprocessing.py",
    "src/feature_engineering.py",
    "src/model_training.py",
    "src/model_evaluation.py",
    "src/utils.py",
    "src/config/loader.py",
    "app/main.py",
    "params.yaml",
    "dvc.yaml",
    "requirements.txt",
    "README.md"
]

def create_structure():
    print("\n🚀 Creating E-Commerce MLOps Project Structure...\n")

    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"[DIR]  {folder}")

    for file in files:
        with open(file, "w") as f:
            f.write("")   # empty file
        print(f"[FILE] {file}")

    print("\n✅ Project structure created successfully!\n")

if __name__ == "__main__":
    create_structure()
