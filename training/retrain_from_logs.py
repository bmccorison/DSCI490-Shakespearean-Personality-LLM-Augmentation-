# training/retrain_from_logs.py
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def retrain():
    print("Starting LoRA retraining from conversation logs...")
    subprocess.run([
        "python", str(REPO_ROOT / "training" / "lora_5.py"),
        "--input-file", str(REPO_ROOT / "logging"),
    ], check=True)

if __name__ == "__main__":
    retrain()
