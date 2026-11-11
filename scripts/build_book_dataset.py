#!/usr/bin/env python
"""Fetch and prepare the ABJ book dataset used by the public analysis.

Source
------
``abjData::consumo``, from the R package ``abjData`` maintained by the Brazilian
Jurimetrics Association (ABJ). The dataset is a 1,000-row sample of second-degree
consumer cases from TJSP and is used in the book *Metodologia de Pesquisa
Jurimétrica* by Ricardo Feliz and Julio Trecenti (ABJ, 2022).

Both the book and the package are MIT licensed. See ``CREDITS.md`` and
``src/claimvalue/data/book/SOURCE.md``.

The data is written to ``data/raw/book/``, which is gitignored. It is never
committed to the repository.

Usage
-----
    pip install pyreadr
    python scripts/build_book_dataset.py
    python scripts/build_book_dataset.py --rda /path/to/consumo.rda --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from claimvalue.book_data import fetch_book_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rda", type=Path, help="local consumo.rda; omit to download the pinned file")
    parser.add_argument("--out-dir", type=Path, help="output directory (default: data/raw/book)")
    parser.add_argument("--force", action="store_true", help="re-download and overwrite")
    args = parser.parse_args()

    result = fetch_book_data(rda_path=args.rda, out_dir=args.out_dir, force=args.force)
    print(f"wrote {result['csv']} ({result['rows']} rows, cached={result['cached']})")
    print(f"source sha256: {result['sha256']}")


if __name__ == "__main__":
    main()
