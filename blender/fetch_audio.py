#!/usr/bin/env python3
"""Download the audio a blend's manifests point to, at the exact commit each source was pinned to.

  pip install huggingface_hub
  HF_TOKEN=hf_... python fetch_audio.py          # token only needed for gated sources

Files land under audio/<owner>__<name>/data/<split>/..., the paths the manifests already use.
Re-running skips files already downloaded.
"""
import json
from collections import defaultdict
from pathlib import Path

from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent
want = defaultdict(set)  # (repo, sha) -> paths inside the repo
for m in (HERE / "manifests").glob("*.jsonl"):
    for line in open(m, encoding="utf-8"):
        r = json.loads(line)
        repo, rest = r["audio_url"].split("/datasets/", 1)[1].split("/resolve/", 1)
        sha, path = rest.split("/", 1)
        want[(repo, sha)].add(path)

for (repo, sha), paths in want.items():
    dest = HERE / "audio" / repo.replace("/", "__")
    print(f"{repo}@{sha[:12]}: {len(paths)} files")
    snapshot_download(repo, repo_type="dataset", revision=sha, local_dir=dest, allow_patterns=sorted(paths))
print("done")
