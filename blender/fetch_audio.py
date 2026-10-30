#!/usr/bin/env python3
"""Download the audio a blend's manifests point to, at the exact commit each source was pinned to.

  pip install huggingface_hub
  HF_TOKEN=hf_... python fetch_audio.py          # token only needed for gated sources

Files land under audio/<owner>__<name>/data/<split>/..., the paths the manifests already use.
Re-running skips files already downloaded.

A row with `audio_offset` / `audio_length` comes from a Parquet source: its `audio_url` is a shard, and
the audio file is that byte range of it (stored verbatim). The shards holding missing files are downloaded
(to the Hugging Face cache) and the files cut out. Other rows' `audio_url` is the audio file itself.
"""
import json
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

HERE = Path(__file__).resolve().parent
want = defaultdict(set)     # (repo, sha) -> audio files inside the repo
shards = defaultdict(dict)  # (repo, sha, shard) -> {local path: (offset, length)}
for m in (HERE / "manifests").glob("*.jsonl"):
    for line in open(m, encoding="utf-8"):
        r = json.loads(line)
        repo, rest = r["audio_url"].split("/datasets/", 1)[1].split("/resolve/", 1)
        sha, path = rest.split("/", 1)
        if "audio_offset" in r:
            dest = HERE / r["audio_filepath"]
            if not dest.exists() or dest.stat().st_size != r["audio_length"]:
                shards[(repo, sha, path)][dest] = (r["audio_offset"], r["audio_length"])
        else:
            want[(repo, sha)].add(path)

for (repo, sha), paths in want.items():
    dest = HERE / "audio" / repo.replace("/", "__")
    print(f"{repo}@{sha[:12]}: {len(paths)} files")
    snapshot_download(repo, repo_type="dataset", revision=sha, local_dir=dest, allow_patterns=sorted(paths))
for (repo, sha, shard), files in shards.items():
    print(f"{repo}@{sha[:12]} {shard}: {len(files)} files")
    with open(hf_hub_download(repo, shard, repo_type="dataset", revision=sha), "rb") as f:
        for dest, (offset, length) in files.items():
            f.seek(offset)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(f.read(length))
print("done")
