"""Download CUAD and select the contracts worth indexing.

Architecture.md §3 calls for 10-20 contracts rather than the full 510. Which 10-20
matters: the whole thesis is that cross-references and definitional dependency break
naive RAG, so contracts are scored on how much of both they contain and the densest
are kept. A random sample would understate the agentic layer's benefit.

Usage:
    python scripts/fetch_cuad.py                 # download + select 15
    python scripts/fetch_cuad.py --count 20
    python scripts/fetch_cuad.py --zip path/to/CUAD_v1.zip   # skip the download
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

CUAD_URL = "https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / ".cache"

# Scoring signals -- deliberately the same surface the parser keys off.
RE_SECTION_REF = re.compile(r"\b(?:Section|Article|Clause)s?\s+\d|\b§", re.I)
RE_DEFINITION = re.compile(r"[“\"'][A-Z][A-Za-z ]{2,40}[”\"']\s*(?:shall\s+mean|means)")
RE_HEADING = re.compile(r"^\s*(?:ARTICLE\s+[IVXLC0-9]|SECTION\s+\d|\d+\.\d+\s)", re.I | re.M)

MIN_CHARS = 12_000
MAX_CHARS = 260_000


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"Using cached {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    print(f"Downloading CUAD from {url}\n  -> {dest}")
    with urllib.request.urlopen(url) as resp, dest.open("wb") as fh:  # noqa: S310
        total = int(resp.headers.get("Content-Length") or 0)
        read = 0
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            read += len(chunk)
            if total:
                pct = 100 * read / total
                print(f"\r  {read / 1e6:6.1f} / {total / 1e6:.0f} MB ({pct:4.1f}%)", end="")
    print("\n  done.")
    return dest


def extract_texts(zip_path: Path, work_dir: Path) -> list[Path]:
    """Pull only ``full_contract_txt/*.txt`` out of the archive."""
    work_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            n
            for n in zf.namelist()
            if "full_contract_txt/" in n and n.lower().endswith(".txt")
        ]
        if not members:
            raise SystemExit("No full_contract_txt/*.txt found in the archive.")
        for name in members:
            target = work_dir / Path(name).name
            if not target.exists():
                with zf.open(name) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            out.append(target)
    print(f"Extracted {len(out)} contract texts.")
    return out


def score(text: str) -> tuple[int, dict[str, int]]:
    """Rank a contract by structural density. Higher is a better fit for this project."""
    refs = len(RE_SECTION_REF.findall(text))
    defs = len(RE_DEFINITION.findall(text))
    headings = len(RE_HEADING.findall(text))
    # Cross-references are the point, so they carry the most weight.
    total = refs * 3 + defs * 4 + headings
    return total, {"refs": refs, "definitions": defs, "headings": headings}


def select(paths: list[Path], count: int) -> list[tuple[Path, int, dict[str, int]]]:
    scored: list[tuple[Path, int, dict[str, int]]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not MIN_CHARS <= len(text) <= MAX_CHARS:
            continue
        total, parts = score(text)
        if parts["headings"] < 10:
            continue  # unparseable by the rule-based parser; skip rather than pollute
        scored.append((path, total, parts))
    scored.sort(key=lambda row: row[1], reverse=True)
    return scored[:count]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=15, help="contracts to keep (default 15)")
    ap.add_argument("--zip", type=Path, default=None, help="local CUAD_v1.zip")
    args = ap.parse_args()

    zip_path = args.zip or download(CUAD_URL, CACHE_DIR / "CUAD_v1.zip")
    texts = extract_texts(zip_path, CACHE_DIR / "full_contract_txt")
    chosen = select(texts, args.count)
    if not chosen:
        raise SystemExit("No contracts passed the selection filter.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for existing in RAW_DIR.glob("doc_*.txt"):
        existing.unlink()

    manifest = []
    for i, (path, total, parts) in enumerate(chosen, start=1):
        doc_id = f"doc_{i:03d}"
        text = path.read_text(encoding="utf-8", errors="ignore")
        (RAW_DIR / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        manifest.append(
            {
                "doc_id": doc_id,
                "source_file": path.name,
                "title": path.stem.replace("_", " "),
                "chars": len(text),
                "density_score": total,
                **parts,
            }
        )
        print(f"{doc_id}  score={total:5d}  refs={parts['refs']:4d}  "
              f"defs={parts['definitions']:3d}  {path.stem[:60]}")

    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {len(manifest)} contracts to {RAW_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
