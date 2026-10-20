"""CLI: build a blend from a recipe file and write the bundle zip.

  python -m blender --recipe recipes/example.json [--out blend.zip]
  python -m blender --scan                       # what each library source holds

The recipe may be a plain recipe or a blend.json from an earlier bundle (its "recipe" is used,
with every source already pinned to a sha). Token: HF_TOKEN or the cached `hf auth login`.
"""
import argparse, json, sys
from pathlib import Path

from huggingface_hub import get_token

from . import core


def main():
    ap = argparse.ArgumentParser(prog="blender")
    ap.add_argument("--recipe")
    ap.add_argument("--out")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--library", default=str(core.ROOT / "library.json"))
    a = ap.parse_args()
    tok = get_token()
    if a.scan:
        lib = core.load_library(a.library)
        print(json.dumps(core.scan(lib["sources"], tok), indent=1))
        return
    if not a.recipe:
        ap.error("--recipe or --scan")
    r = json.loads(Path(a.recipe).read_text())
    r = r.get("recipe", r)
    rows, report = core.blend(r, tok)
    out = a.out or f"{report['recipe']['name']}.zip"
    Path(out).write_bytes(core.bundle(rows, report))
    json.dump({k: report[k] for k in ("splits", "train_mix", "warnings")}, sys.stdout, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
