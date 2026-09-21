#!/usr/bin/env python3
"""Generate a synthetic NCBI-shaped export for testing the pipeline.

This exists so the code can be exercised end to end without the real cohort,
which is not reconstructable yet. The fixture is SYNTHETIC: it is generated
from a random seed with a planted genotype-phenotype relationship, and no
number produced from it may appear in the manuscript or in any results table.
Files are written with a ``SYNTHETIC_`` prefix for that reason.

    python tools/make_synthetic_fixture.py --out tests/fixtures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RESISTANCE_DRIVER = "gyrA=T86I"
BACKGROUND_GENES = ["tetO", "aph(3')-IIIa", "blaOXA-61", "cmeA", "cmeB", "ermB", "aadE"]


def build(n_isolates: int = 600, n_clusters: int = 60, seed: int = 20260921) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cluster = rng.integers(0, n_clusters, size=n_isolates)

    # Resistance is driven by the planted determinant, with cluster-level
    # correlation so that grouped CV has something to protect against.
    cluster_effect = rng.normal(0, 1.0, size=n_clusters)[cluster]
    carries_driver = rng.random(n_isolates) < 1 / (1 + np.exp(-(cluster_effect - 0.8)))

    rows = []
    for i in range(n_isolates):
        genes = list(rng.choice(BACKGROUND_GENES, size=rng.integers(0, 4), replace=False))
        if carries_driver[i]:
            genes.append(RESISTANCE_DRIVER)
        # 5% label noise, so a perfect score is not attainable by construction.
        resistant = bool(carries_driver[i] ^ (rng.random() < 0.05))
        ast = [f"ciprofloxacin={'R' if resistant else 'S'}"]
        if rng.random() < 0.6:
            ast.append(f"tetracycline={'R' if 'tetO' in genes else 'S'}")
        if rng.random() < 0.1:
            ast.append("oxytetracycline=S")  # decoy for substring matching
        rows.append(
            {
                "#Organism group": "Campylobacter jejuni",
                "Isolate": f"SYN{i:05d}",
                "BioSample": f"SAMNSYN{i:05d}",
                "Assembly": f"GCA_SYN{i:05d}",
                "SNP cluster": f"PDS000{cluster[i]:05d}.1",
                "Isolation type": rng.choice(["clinical", "environmental/other"]),
                "Location": rng.choice(["USA", "United Kingdom", "Australia"]),
                "Create date": f"20{rng.integers(15, 25):02d}-0{rng.integers(1, 10)}-01",
                "AMR genotypes": ",".join(sorted(genes)) if genes else None,
                "AST phenotypes": ",".join(ast),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("tests/fixtures"))
    parser.add_argument("--n-isolates", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    frame = build(n_isolates=args.n_isolates, seed=args.seed)
    path = args.out / "SYNTHETIC_Campylobacter_jejuni.csv"
    frame.to_csv(path, index=False)
    print(f"Wrote {len(frame)} synthetic rows to {path}")
    print("SYNTHETIC DATA -- for code testing only, never for reported results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
