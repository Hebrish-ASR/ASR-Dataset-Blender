"""Blend Hugging Face audio datasets into one training set: pick sources, filter rows, hold out
validation and test, size each source (all, a minutes cap, or a share of the total), and write
NeMo manifests plus a reproducible recipe. Reads only metadata.jsonl, never audio.

Source layout (the Hugging Face "audiofolder" layout every dataset in library.json uses):
  data/<split>/metadata.jsonl   one JSON row per clip: file_name, text, duration_s, id, ...
  data/<split>/<file_name>      the audio, fetched later by the bundle's fetch_audio.py
"""
import hashlib, io, json, random, zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, GatedRepoError, RepositoryNotFoundError

ROOT = Path(__file__).resolve().parent.parent
_CACHE = {}  # (repo, sha, split) -> rows; a sha is immutable, so this never goes stale


class AccessError(Exception):
    """The source is gated or private and the token cannot read it."""


def load_library(path=ROOT / "library.json"):
    return json.loads(Path(path).read_text())


def h(s):
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def mins(rows):
    return round(sum(r["duration_s"] for r in rows) / 60, 1)


def resolve(repo, revision="main", token=None):
    """Pin a revision to a commit sha and list the splits that have a metadata.jsonl."""
    api = HfApi(token=token)
    try:
        sha = api.dataset_info(repo, revision=revision).sha
        try:
            tree = list(api.list_repo_tree(repo, path_in_repo="data", repo_type="dataset", revision=sha))
        except EntryNotFoundError:
            tree = []  # created but nothing published yet
        splits = sorted(p.path.split("/")[1] for p in tree if p.path.count("/") == 1 and "." not in p.path.split("/")[1])
    except (GatedRepoError, RepositoryNotFoundError) as e:
        raise AccessError(f"{repo}: no access ({type(e).__name__}). If it is gated, request access at "
                          f"https://huggingface.co/datasets/{repo} and sign in.") from e
    return sha, splits


def load_rows(repo, sha, split, token=None):
    key = (repo, sha, split)
    if key not in _CACHE:
        try:
            p = hf_hub_download(repo, f"data/{split}/metadata.jsonl", repo_type="dataset", revision=sha, token=token)
        except GatedRepoError as e:
            raise AccessError(f"{repo}: gated; request access at https://huggingface.co/datasets/{repo}") from e
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        for r in rows:
            r.setdefault("id", Path(r["file_name"]).stem)
        _CACHE[key] = rows
    return _CACHE[key]


def scan(sources, token=None):
    """Per source: pinned sha, splits, rows, minutes, and the categories and speakers present.
    `sources` is a list of {"repo", "revision"}; returns {repo: info} (or {"error": ...})."""
    out = {}
    for s in sources:
        repo = s["repo"]
        try:
            sha, splits = resolve(repo, s.get("revision", "main"), token)
            info = {"sha": sha, "splits": {}, "categories": defaultdict(float), "speakers": defaultdict(float)}
            for sp in splits:
                rows = load_rows(repo, sha, sp, token)
                info["splits"][sp] = {"n": len(rows), "min": mins(rows)}
                for r in rows:
                    for c in r.get("categories") or []:
                        info["categories"][c] += r["duration_s"] / 60
                    info["speakers"][r.get("speaker") or ""] += r["duration_s"] / 60
            info["categories"] = {k: round(v, 1) for k, v in sorted(info["categories"].items())}
            info["speakers"] = {k: round(v, 1) for k, v in sorted(info["speakers"].items())}
            out[repo] = info
        except AccessError as e:
            out[repo] = {"error": str(e)}
    return out


# ---------------------------------------------------------------- the blend

DEFAULT_RECIPE = {
    "name": "blend",
    "sources": [],              # [{"repo", "revision": "main", "share": null | 0-1, "cap_min": null | minutes}]
    "filters": {
        "include_categories": [],   # keep a row only if it has at least one of these (empty = keep all)
        "exclude_categories": [],   # drop a row that has any of these
        "speakers": [],             # keep only these speakers (empty = all); "" matches rows with no speaker field
        "default_speaker": "",      # the speaker a row with no `speaker` field is taken to be
        "min_duration_s": 0, "max_duration_s": 0,  # 0 = no limit
    },
    "dedupe": True,             # an id that appears in several sources is kept once, from the first source listed
    "held_out": {
        "mode": "source",           # "source": keep each source's test split as test; "resplit": pool everything and split anew
        "val_pct": 6.25,            # share of train carved off for validation (checkpoint selection)
        "test_pct": 10.0,           # resplit only: share that becomes test
        "group_by": "id",           # "id", or "target_terms": hold out terms, not rows (no held-out term in train)
    },
    "oversample": False,        # a share larger than its source repeats rows instead of falling short
    "text_field": "text",
    "seed": 7,
}


def merge_recipe(r):
    out = json.loads(json.dumps(DEFAULT_RECIPE))
    for k, v in (r or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def _keep(r, f):
    cats = set(r.get("categories") or [])
    if f["include_categories"] and not cats & set(f["include_categories"]):
        return False
    if cats & set(f["exclude_categories"]):
        return False
    if f["speakers"] and (r.get("speaker") or f["default_speaker"]) not in f["speakers"]:
        return False
    d = r["duration_s"]
    return d >= (f["min_duration_s"] or 0) and (not f["max_duration_s"] or d <= f["max_duration_s"])


def _bucket(key, seed):
    return h(f"{seed}:{key}") % 10000 / 100  # 0-100, stable for a key and seed


def _take(rows, want_min, rng, oversample):
    """Rows up to `want_min` minutes, in seeded random order; repeat the pool when oversampling."""
    pool = rows[:]
    if not sum(r["duration_s"] for r in pool):
        return []
    rng.shuffle(pool)
    out, got, rep = [], 0.0, 0
    while pool:
        for r in pool:
            if got >= want_min * 60:
                return out
            out.append(dict(r, _repeat=rep) if rep else r)
            got += r["duration_s"]
        if not oversample:
            break
        rep += 1
    return out


def blend(recipe, token=None):
    """Returns (rows_by_split, report). Rows carry `_source` and `_split_src` for the manifests."""
    R = merge_recipe(recipe)
    f, ho = R["filters"], R["held_out"]
    if not R["sources"]:
        raise ValueError("no sources selected")
    seen, pooled, report = set(), [], {"sources": {}, "warnings": []}
    for s in R["sources"]:
        repo = s["repo"]
        sha, splits = resolve(repo, s.get("revision", "main"), token)
        s["revision"] = sha  # pinned in the recipe that is written out
        st = {"sha": sha, "read": 0, "filtered_out": 0, "duplicates_dropped": 0}
        for sp in splits:
            for r in load_rows(repo, sha, sp, token):
                st["read"] += 1
                if not _keep(r, f):
                    st["filtered_out"] += 1
                    continue
                if R["dedupe"] and r["id"] in seen:
                    st["duplicates_dropped"] += 1
                    continue
                seen.add(r["id"])
                pooled.append(dict(r, _source=repo, _sha=sha, _split_src=sp, _key=f"{repo}:{r['id']}"))
        report["sources"][repo] = st

    # Held out first, so validation and test stay the same whatever the caps and shares do to train.
    # group_by "target_terms" holds out TERMS: a held-out term sends every row containing it to
    # that split, so no held-out term is ever in train (the rule the Hebrish sentence bank uses;
    # a row's other, non-held-out terms may appear in train). Rows with no terms split by id.
    seed = R["seed"]
    terms = lambda r: [t.casefold() for t in (r.get("target_terms") or [])] if ho["group_by"] == "target_terms" else []
    tb = lambda t: _bucket("term:" + t, seed)
    split_of = {}
    for r in pooled:
        ts, b = terms(r), _bucket(r["id"], seed)
        if ho["mode"] == "resplit":
            if ts:
                split_of[r["_key"]] = ("test" if any(tb(t) < ho["test_pct"] for t in ts) else
                                       "val" if any(tb(t) < ho["test_pct"] + ho["val_pct"] for t in ts) else "train")
            else:
                split_of[r["_key"]] = "test" if b < ho["test_pct"] else "val" if b < ho["test_pct"] + ho["val_pct"] else "train"
        elif r["_split_src"] != "train":
            split_of[r["_key"]] = "test"  # the source's own test (or other non-train) split, as published
        elif ts:
            split_of[r["_key"]] = "val" if any(tb(t) < ho["val_pct"] for t in ts) else "train"
        else:
            split_of[r["_key"]] = "val" if b < ho["val_pct"] else "train"

    by = defaultdict(lambda: defaultdict(list))
    for r in pooled:
        by[split_of[r["_key"]]][r["_source"]].append(r)

    # Size train per source: shares are fractions of the train total; the rest give all rows or a cap.
    srcs = {s["repo"]: s for s in R["sources"]}
    avail = {repo: mins(by["train"][repo]) for repo in srcs}
    fixed = {repo: min(avail[repo], s.get("cap_min") or avail[repo]) for repo, s in srcs.items() if not s.get("share")}
    shares = {repo: float(s["share"]) for repo, s in srcs.items() if s.get("share")}
    P = sum(shares.values())
    if shares and fixed and P >= 1:
        raise ValueError("shares add up to 100% or more but other sources have no share; give every source a share, or lower them")
    if shares and not fixed:
        shares = {k: v / P for k, v in shares.items()}  # all sources have shares: they are relative weights
        # Largest total the shares allow; with oversampling, the largest source sets it instead.
        total = (max if R["oversample"] else min)(avail[k] / v for k, v in shares.items() if v)
    else:
        total = sum(fixed.values()) / (1 - P) if shares else sum(fixed.values())
    want = {**fixed, **{k: total * v for k, v in shares.items()}}

    rng = random.Random(seed)
    train, mix = [], {}
    for repo in srcs:  # listed order, so the same recipe gives the same rows
        rows = _take(by["train"][repo], want[repo], rng, R["oversample"] and repo in shares)
        train += rows
        got = mins(rows)
        mix[repo] = {"available_min": avail[repo], "wanted_min": round(want[repo], 1), "got_min": got}
        if repo in shares and got < want[repo] - 0.5:
            report["warnings"].append(f"{repo}: wanted {want[repo]:.1f} min, has {avail[repo]} (turn on oversample to repeat rows)")
        if avail[repo] and got / avail[repo] > 3:
            report["warnings"].append(f"{repo}: oversampled {got / avail[repo]:.0f}x ({avail[repo]} min repeated to {got})")
    tot = sum(m["got_min"] for m in mix.values()) or 1
    for m in mix.values():
        m["share"] = round(m["got_min"] / tot, 3)
    rng.shuffle(train)

    out = {"train": train,
           "val": [r for repo in srcs for r in by["val"][repo]],
           "test": [r for repo in srcs for r in by["test"][repo]]}
    report["train_mix"] = mix
    report["splits"] = {k: {"n": len(v), "min": mins(v)} for k, v in out.items()}
    report["recipe"] = R
    return out, report


# ---------------------------------------------------------------- the bundle

def _local_path(r):
    return f"audio/{r['_source'].replace('/', '__')}/data/{r['_split_src']}/{r['file_name']}"


def manifest_row(r, text_field):
    row = {"audio_filepath": _local_path(r), "duration": r["duration_s"],
           "text": r.get(text_field) or r.get("text", ""), "id": r["id"], "source": r["_source"],
           "audio_url": f"https://huggingface.co/datasets/{r['_source']}/resolve/{r['_sha']}/data/{r['_split_src']}/{r['file_name']}"}
    if r.get("_repeat"):
        row["repeat"] = r["_repeat"]
    for k, v in r.items():  # keep the source's own metadata (categories, speaker, target_terms, ...)
        if not k.startswith("_") and k not in row and k not in ("file_name", "duration_s"):
            row[k] = v
    return row


def _yaml_list(items):
    """input_cfg.yaml without a YAML dependency: a list of flat maps with scalar values."""
    lines = []
    for it in items:
        for i, (k, v) in enumerate(it.items()):
            lines.append(("- " if i == 0 else "  ") + f"{k}: {json.dumps(v)}")
    return "\n".join(lines) + "\n"


def bundle(rows, report):
    """A zip: blend.json (recipe + report), manifests/, input_cfg.yaml, fetch_audio.py, README.md."""
    R = report["recipe"]
    tf = R["text_field"]
    buf = io.BytesIO()
    z = zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED)
    jl = lambda rs: "".join(json.dumps(manifest_row(r, tf), ensure_ascii=False) + "\n" for r in rs)
    for split, rs in rows.items():
        z.writestr(f"manifests/{split}.jsonl", jl(rs))
    per_src = defaultdict(list)
    for r in rows["train"]:
        per_src[r["_source"]].append(r)
    cfg = []
    for repo, rs in per_src.items():
        name = repo.replace("/", "__")
        z.writestr(f"manifests/train_by_source/{name}.jsonl", jl(rs))
        cfg.append({"type": "nemo", "manifest_filepath": f"manifests/train_by_source/{name}.jsonl",
                    "weight": report["train_mix"][repo]["share"], "tags": {"source": repo}})
    z.writestr("input_cfg.yaml", "# NeMo Lhotse input_cfg: the same mix as manifests/train.jsonl, as weighted sources.\n"
               "# Weights are each source's share of train minutes in this blend.\n" + _yaml_list(
                   [{k: v for k, v in c.items() if k != "tags"} for c in cfg]))
    z.writestr("blend.json", json.dumps({"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                         "tool": "https://github.com/Hebrish-ASR/ASR-Dataset-Blender",
                                         **report}, indent=1, ensure_ascii=False))
    z.writestr("fetch_audio.py", (ROOT / "blender" / "fetch_audio.py").read_text())
    s = report["splits"]
    z.writestr("README.md", f"""# {R['name']}

Made by [ASR Dataset Blender](https://github.com/Hebrish-ASR/ASR-Dataset-Blender). Metadata only; get the audio with

    pip install huggingface_hub && HF_TOKEN=hf_... python fetch_audio.py

(the token is needed for gated sources). Paths in `manifests/*.jsonl` are relative to this folder.

| split | rows | minutes |
|---|---|---|
""" + "".join(f"| {k} | {v['n']} | {v['min']} |\n" for k, v in s.items()) + """
`blend.json` holds the recipe with every source pinned to a commit sha; load it in the Blender (or
`python -m blender --recipe blend.json`) to rebuild exactly this blend.
""")
    z.close()
    return buf.getvalue()
