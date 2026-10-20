#!/usr/bin/env python3
"""Deploy this repo to the Hugging Face Space (created public, free cpu-basic, on first run).

  uv run --with huggingface_hub python scripts/deploy_space.py
"""
from pathlib import Path
from huggingface_hub import HfApi

SPACE = "danielrosehill/ASR-Dataset-Blender"
ROOT = Path(__file__).resolve().parent.parent
api = HfApi()
api.create_repo(SPACE, repo_type="space", space_sdk="gradio", exist_ok=True)
api.upload_folder(repo_id=SPACE, repo_type="space", folder_path=ROOT,
                  allow_patterns=["app.py", "README.md", "requirements.txt", "library.json", "blender/*.py", "recipes/*.json"],
                  commit_message="Deploy from github.com/Hebrish-ASR/ASR-Dataset-Blender")
print(f"https://huggingface.co/spaces/{SPACE}")
