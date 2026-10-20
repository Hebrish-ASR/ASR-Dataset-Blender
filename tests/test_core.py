"""Offline tests: synthetic sources stand in for the Hub (resolve and load_rows are patched)."""
import io, json, zipfile

import pytest

from blender import core

SRC = {
    "a/cs": {"train": [{"id": f"cs{i}", "file_name": f"audio/cs{i}.wav", "text": f"cs {i}", "duration_s": 6.0,
                        "categories": ["code_switch"], "target_terms": [f"term{i % 40}", "shared"]} for i in range(400)],
             "test": [{"id": f"cst{i}", "file_name": f"audio/cst{i}.wav", "text": "t", "duration_s": 6.0,
                       "categories": ["code_switch"], "target_terms": ["held"]} for i in range(20)]},
    "a/en": {"train": [{"id": f"en{i}", "file_name": f"audio/en{i}.wav", "text": f"en {i}", "duration_s": 3.0,
                        "categories": ["general"], "speaker": "Other" if i % 10 == 0 else None} for i in range(100)]
             + [{"id": "cs0", "file_name": "audio/cs0.wav", "text": "dup", "duration_s": 6.0, "categories": ["general"]}]},
}


@pytest.fixture(autouse=True)
def fake_hub(monkeypatch):
    monkeypatch.setattr(core, "resolve", lambda repo, revision="main", token=None: ("f" * 40, sorted(SRC[repo])))
    monkeypatch.setattr(core, "load_rows", lambda repo, sha, split, token=None: SRC[repo][split])


def test_all_rows_and_source_test_split():
    rows, rep = core.blend({"sources": [{"repo": "a/cs"}, {"repo": "a/en"}], "held_out": {"val_pct": 0}})
    assert rep["splits"]["test"]["n"] == 20
    assert rep["splits"]["train"]["n"] == 400 + 100  # the duplicate cs0 in a/en is dropped
    assert rep["sources"]["a/en"]["duplicates_dropped"] == 1


def test_share_sizes_against_fixed_sources():
    rows, rep = core.blend({"sources": [{"repo": "a/cs"}, {"repo": "a/en", "share": 0.1}], "held_out": {"val_pct": 0}})
    m = rep["train_mix"]
    assert m["a/cs"]["got_min"] == 40.0
    assert abs(m["a/en"]["wanted_min"] - 40 / 0.9 * 0.1) < 0.1
    assert abs(m["a/en"]["share"] - 0.1) < 0.01


def test_short_share_warns_and_oversample_fills():
    r = {"sources": [{"repo": "a/cs"}, {"repo": "a/en", "share": 0.5}], "held_out": {"val_pct": 0}}
    _, rep = core.blend(r)
    assert rep["train_mix"]["a/en"]["got_min"] == 5.0 and rep["warnings"]
    rows, rep = core.blend({**r, "oversample": True})
    assert rep["train_mix"]["a/en"]["got_min"] >= 39.9
    assert any(x.get("_repeat") for x in rows["train"])


def test_cap_and_filters():
    _, rep = core.blend({"sources": [{"repo": "a/cs", "cap_min": 10}], "held_out": {"val_pct": 0}})
    assert rep["train_mix"]["a/cs"]["got_min"] == 10.0
    _, rep = core.blend({"sources": [{"repo": "a/en"}], "held_out": {"val_pct": 0},
                         "filters": {"speakers": ["Daniel"], "default_speaker": "Daniel"}})
    assert rep["splits"]["train"]["n"] == 91  # 10 rows by "Other" filtered out


def test_term_holdout_never_trains_on_a_held_out_term():
    rows, _ = core.blend({"sources": [{"repo": "a/cs"}],
                          "held_out": {"mode": "resplit", "group_by": "target_terms", "val_pct": 10, "test_pct": 10}})
    held = {t for s in ("val", "test") for r in rows[s] for t in r["target_terms"]
            if core._bucket("term:" + t, 7) < 20}
    assert held and not {t for r in rows["train"] for t in r["target_terms"]} & held


def test_same_recipe_same_blend():
    r = {"sources": [{"repo": "a/cs", "cap_min": 15}, {"repo": "a/en", "share": 0.2}]}
    a, _ = core.blend(r)
    b, _ = core.blend(r)
    assert [x["id"] for x in a["train"]] == [x["id"] for x in b["train"]]


def test_bundle_contents():
    rows, rep = core.blend({"name": "x", "sources": [{"repo": "a/cs"}, {"repo": "a/en", "share": 0.2}]})
    z = zipfile.ZipFile(io.BytesIO(core.bundle(rows, rep)))
    names = set(z.namelist())
    assert {"blend.json", "input_cfg.yaml", "fetch_audio.py", "README.md", "manifests/train.jsonl"} <= names
    first = json.loads(z.read("manifests/train.jsonl").splitlines()[0])
    assert first["audio_filepath"].startswith("audio/a__") and "/resolve/" + "f" * 40 in first["audio_url"]
    assert json.loads(z.read("blend.json"))["recipe"]["sources"][0]["revision"] == "f" * 40
