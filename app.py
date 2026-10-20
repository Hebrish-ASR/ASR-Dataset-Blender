"""Gradio front end for blender.core (the Hugging Face Space). Runs on free CPU: it reads only
metadata.jsonl files and never touches audio, so a blend costs nothing to whoever hosts it.

Gated sources are read with the visitor's own token from "Sign in with Hugging Face"
(hf_oauth in README.md), never with a token stored in the Space, so gating still decides who
gets which rows. Run locally with `python app.py`; there it falls back to HF_TOKEN / the cached login.
"""
import json, os, tempfile
from pathlib import Path

import gradio as gr
from huggingface_hub import get_token

from blender import core

LIB = core.load_library()
ON_SPACE = bool(os.environ.get("SPACE_ID"))
COLS = ["use", "dataset", "share %", "cap min", "train min", "test min", "status"]


def token_of(oauth_token):
    if oauth_token is not None:
        return oauth_token.token
    return None if ON_SPACE else get_token()


def initial_table():
    return [[i == 0, s["repo"], None, None, None, None, "gated: sign in" if s.get("gated") else ""]
            for i, s in enumerate(LIB["sources"])]


def rows_of(table):
    return table.values.tolist() if hasattr(table, "values") else table


def scan(table, oauth_token: gr.OAuthToken | None = None):
    tok = token_of(oauth_token)
    rows = rows_of(table)
    info = core.scan([{"repo": r[1]} for r in rows], tok)
    cats, spk = {}, {}
    for r in rows:
        i = info[r[1]]
        if "error" in i:
            r[4:7] = [None, None, "no access: request it on the dataset page, then sign in"]
            continue
        sp = i["splits"]
        r[4] = sp.get("train", {}).get("min")
        r[5] = sum(v["min"] for k, v in sp.items() if k != "train")
        r[6] = "ok @ " + i["sha"][:8]
        for c, m in i["categories"].items():
            cats[c] = max(cats.get(c, 0), m)
        for s, m in i["speakers"].items():
            k = s or LIB.get("default_speaker", "")
            spk[k] = max(spk.get(k, 0), m)
    cat_choices = sorted(cats, key=lambda c: -cats[c])
    spk_choices = sorted(spk)
    summary = "Categories (largest source's minutes): " + ", ".join(f"`{c}` {cats[c]:.0f}" for c in cat_choices)
    return (gr.update(value=rows), gr.update(choices=cat_choices), gr.update(choices=cat_choices),
            gr.update(choices=spk_choices), summary)


def num(x):
    try:
        return None if x is None or x == "" or x != x else float(x)  # x != x: NaN from an empty cell
    except (TypeError, ValueError):
        return None


def recipe_from_form(name, table, inc, exc, speakers, dmin, dmax, dedupe, mode, val_pct, test_pct,
                     group_by, oversample, text_field, seed):
    sources = []
    for r in rows_of(table):
        if not r[0] or str(r[0]).lower() == "false":
            continue
        s = {"repo": r[1]}
        if num(r[2]):
            s["share"] = num(r[2]) / 100
        if num(r[3]):
            s["cap_min"] = num(r[3])
        sources.append(s)
    return {"name": name or "blend", "sources": sources,
            "filters": {"include_categories": inc or [], "exclude_categories": exc or [],
                        "speakers": speakers or [], "default_speaker": LIB.get("default_speaker", ""),
                        "min_duration_s": dmin or 0, "max_duration_s": dmax or 0},
            "dedupe": dedupe,
            "held_out": {"mode": mode, "val_pct": val_pct, "test_pct": test_pct, "group_by": group_by},
            "oversample": oversample, "text_field": text_field or "text", "seed": int(seed or 0)}


def run(recipe, tok):
    try:
        rows, report = core.blend(recipe, tok)
    except (core.AccessError, ValueError) as e:
        raise gr.Error(str(e))
    path = Path(tempfile.mkdtemp()) / f"{report['recipe']['name']}.zip"
    path.write_bytes(core.bundle(rows, report))
    md = ["| split | rows | minutes |", "|---|---|---|"]
    md += [f"| {k} | {v['n']} | {v['min']} |" for k, v in report["splits"].items()]
    md += ["", "| train source | available | wanted | got | share |", "|---|---|---|---|---|"]
    md += [f"| {k} | {v['available_min']} | {v['wanted_min']} | {v['got_min']} | {v['share']:.1%} |"
           for k, v in report["train_mix"].items()]
    md += [f"\n**Warning:** {w}" for w in report["warnings"]]
    return "\n".join(md), str(path), json.dumps(report["recipe"], indent=1)


def build(name, table, inc, exc, speakers, dmin, dmax, dedupe, mode, val_pct, test_pct, group_by,
          oversample, text_field, seed, oauth_token: gr.OAuthToken | None = None):
    recipe = recipe_from_form(name, table, inc, exc, speakers, dmin, dmax, dedupe, mode, val_pct, test_pct,
                              group_by, oversample, text_field, seed)
    return run(recipe, token_of(oauth_token))


def rebuild(file, oauth_token: gr.OAuthToken | None = None):
    if not file:
        raise gr.Error("upload a recipe or a blend.json first")
    r = json.loads(Path(file).read_text())
    return run(r.get("recipe", r), token_of(oauth_token))


with gr.Blocks(title="ASR Dataset Blender") as demo:
    gr.Markdown(f"""# ASR Dataset Blender
Blend speech datasets from the **{LIB['name']}** library into one training set and download it:
NeMo manifests for train / val / test, a NeMo Lhotse `input_cfg.yaml`, a `blend.json` recipe with every
source pinned to a commit, and `fetch_audio.py` to pull the audio. Only metadata is processed here; the
audio downloads on your machine. [Source and CLI](https://github.com/Hebrish-ASR/ASR-Dataset-Blender) ·
[collection]({LIB['collection']})

Two of the datasets are gated: request access on their pages, then **sign in** so the Blender can read
them as you.""")
    gr.LoginButton()
    with gr.Tab("Blend"):
        gr.Markdown("**1. Sources.** Tick the datasets to use. Leave *share %* and *cap min* empty to take every "
                    "row. *cap min* limits a source to that many train minutes. *share %* makes the source that "
                    "fraction of train minutes; when every ticked source has a share, they are relative weights. "
                    "Click **Scan** to see sizes and fill the category and speaker lists.")
        table = gr.Dataframe(value=initial_table(), headers=COLS,
                             datatype=["bool", "str", "number", "number", "number", "number", "str"],
                             interactive=True, column_count=(len(COLS), "fixed"))
        scan_btn = gr.Button("Scan sources")
        cat_md = gr.Markdown()
        gr.Markdown("**2. Filters**")
        with gr.Row():
            inc = gr.Dropdown([], multiselect=True, allow_custom_value=True, label="Only rows with any of these categories")
            exc = gr.Dropdown([], multiselect=True, allow_custom_value=True, label="Drop rows with any of these categories")
        with gr.Row():
            speakers = gr.Dropdown([], multiselect=True, allow_custom_value=True, label="Speakers (empty = all)")
            dmin = gr.Number(0, label="Min clip seconds (0 = none)")
            dmax = gr.Number(0, label="Max clip seconds (0 = none)")
        dedupe = gr.Checkbox(True, label="Drop duplicate ids across sources (keep the first source's copy)")
        gr.Markdown("**3. Held-out data.** Validation and test are chosen before the mix is sized, so caps and "
                    "shares never change them.")
        with gr.Row():
            mode = gr.Radio([("Keep each source's own test split", "source"), ("Pool everything and split anew", "resplit")],
                            value="source", label="Test set")
            val_pct = gr.Slider(0, 30, 6.25, step=0.25, label="Validation % (of train)")
            test_pct = gr.Slider(0, 30, 10, step=0.5, label="Test % (resplit only)")
        group_by = gr.Radio([("Row id", "id"), ("Target terms (term-disjoint)", "target_terms")], value="id",
                            label="Hold out by: with target terms, a held-out term sends every row containing it to val/test, so it is never trained on")
        gr.Markdown("**4. Output**")
        with gr.Row():
            name = gr.Textbox("blend", label="Blend name")
            text_field = gr.Textbox("text", label="Transcript field")
            seed = gr.Number(7, precision=0, label="Seed")
            oversample = gr.Checkbox(False, label="Oversample a share its source cannot fill (repeats rows)")
        go = gr.Button("Build blend", variant="primary")
    with gr.Tab("Rebuild from recipe"):
        gr.Markdown("Upload a `blend.json` from an earlier bundle (or a recipe file) to rebuild exactly that blend.")
        up = gr.File(file_types=[".json"], type="filepath")
        go2 = gr.Button("Rebuild", variant="primary")
    report = gr.Markdown()
    out = gr.File(label="Download")
    with gr.Accordion("Recipe used", open=False):
        recipe_box = gr.Code(language="json")

    scan_btn.click(scan, [table], [table, inc, exc, speakers, cat_md])
    go.click(build, [name, table, inc, exc, speakers, dmin, dmax, dedupe, mode, val_pct, test_pct, group_by,
                     oversample, text_field, seed], [report, out, recipe_box])
    go2.click(rebuild, [up], [report, out, recipe_box])

if __name__ == "__main__":
    demo.launch()
