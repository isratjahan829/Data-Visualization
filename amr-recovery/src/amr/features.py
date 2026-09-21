"""Versioned feature dictionary for AMR determinant presence/absence.

Deliverable D3. Two defects in the original pipeline are addressed here.

1. *Selection leakage.* The gene vocabulary was built by counting tokens over
   the whole file and keeping the top 2,000, before any train/test split. The
   vocabulary is a fitted preprocessing artefact, so it must be learned from
   training rows only. ``GeneVocabulary`` is therefore a fit/transform object
   and the pipeline fits it inside each fold.

2. *Undocumented tokenisation.* One code path split on ``[;,]`` and kept the
   text left of ``=``; another additionally split on ``=`` and ``:`` and
   dropped tokens of length <= 2. The two produce different feature spaces
   from the same cell, which is how 147, 205, 208 and 542 came to describe
   "the" feature count. Tokenisation is defined once, here, and every feature
   carries its determinant class in the exported dictionary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["GeneVocabulary", "tokenise_amr_cell", "FEATURE_KINDS"]

FEATURE_KINDS = ("gene", "point_mutation")

_SPLIT = re.compile(r"[,;|]")
# NCBI AMRFinderPlus reports point mutations as ``gene=SUBSTITUTION``
# (e.g. ``gyrA=T86I``). A bare symbol (``tetO``) is an acquired gene.
_POINT = re.compile(r"^(?P<gene>[^=]+)=(?P<substitution>[A-Za-z]\d+[A-Za-z*]+)$")


def tokenise_amr_cell(cell: object) -> list[tuple[str, str]]:
    """Tokenise one 'AMR genotypes' cell into ``(feature_kind, token)`` pairs.

    A point mutation yields two features: the allele (``gyrA_T86I``) and the
    gene it sits in (``gyrA``). Both are emitted deliberately -- the allele is
    the mechanistic determinant, the gene is the coarser signal -- and the
    exported dictionary distinguishes them, so a model can be restricted to
    one kind without re-tokenising.
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    text = str(cell).strip()
    if not text or text.lower() in {"nan", "none", "null", "-"}:
        return []

    features: list[tuple[str, str]] = []
    for raw in _SPLIT.split(text):
        item = raw.strip()
        if not item or item.lower() in {"nan", "none"}:
            continue
        match = _POINT.match(item)
        if match:
            gene = match.group("gene").strip()
            substitution = match.group("substitution").strip().upper()
            features.append(("point_mutation", f"{gene}_{substitution}"))
            features.append(("gene", gene))
        else:
            # ``gene=COMPLETE``/``=PARTIAL`` style qualifiers: keep the symbol.
            symbol = item.split("=", 1)[0].strip()
            if symbol:
                features.append(("gene", symbol))
    # Presence/absence: a determinant listed twice in one cell is still one
    # feature for that isolate.
    return sorted(set(features))


@dataclass
class GeneVocabulary:
    """Fit/transform presence-absence encoder with an exportable dictionary.

    Parameters
    ----------
    min_count
        Minimum number of *training* isolates carrying a determinant for it to
        become a feature. Filtering by count rather than by a fixed "top N"
        keeps the feature space a documented function of the data instead of
        an arbitrary cap that silently changed per species.
    include_kinds
        Which feature kinds to retain; defaults to both.
    """

    min_count: int = 5
    include_kinds: tuple[str, ...] = FEATURE_KINDS
    feature_names_: list[str] = field(default_factory=list, init=False)
    dictionary_: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)
    _index: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        unknown = set(self.include_kinds) - set(FEATURE_KINDS)
        if unknown:
            raise ValueError(f"unknown feature kinds: {sorted(unknown)}")
        if self.min_count < 1:
            raise ValueError("min_count must be >= 1")

    def fit(self, cells: pd.Series) -> "GeneVocabulary":
        """Learn the vocabulary from training cells only."""
        counts: dict[tuple[str, str], int] = {}
        for cell in cells:
            for kind, token in tokenise_amr_cell(cell):
                if kind in self.include_kinds:
                    counts[(kind, token)] = counts.get((kind, token), 0) + 1

        kept = [
            (kind, token, n) for (kind, token), n in counts.items() if n >= self.min_count
        ]
        # Deterministic order: frequency then name, so the dictionary and the
        # design matrix are byte-identical across runs with the same input.
        kept.sort(key=lambda row: (-row[2], row[0], row[1]))

        self.dictionary_ = pd.DataFrame(
            [
                {
                    "feature_id": f"{kind}__{token}",
                    "feature_kind": kind,
                    "determinant": token,
                    "n_training_isolates": n,
                    "training_carriage": n / len(cells) if len(cells) else float("nan"),
                }
                for kind, token, n in kept
            ]
        )
        self.feature_names_ = self.dictionary_["feature_id"].tolist()
        self._index = {name: i for i, name in enumerate(self.feature_names_)}
        return self

    def transform(self, cells: pd.Series) -> pd.DataFrame:
        """Encode cells as a 0/1 design matrix over the fitted vocabulary.

        Determinants absent from the training vocabulary are ignored (and
        counted by :meth:`oov_report`), never appended -- a test fold may not
        widen the feature space.
        """
        if not self.feature_names_:
            raise RuntimeError("vocabulary is not fitted; call fit() first")
        matrix = np.zeros((len(cells), len(self.feature_names_)), dtype=np.uint8)
        for row, cell in enumerate(cells):
            for kind, token in tokenise_amr_cell(cell):
                if kind not in self.include_kinds:
                    continue
                column = self._index.get(f"{kind}__{token}")
                if column is not None:
                    matrix[row, column] = 1
        return pd.DataFrame(matrix, columns=self.feature_names_, index=pd.RangeIndex(len(cells)))

    def fit_transform(self, cells: pd.Series) -> pd.DataFrame:
        return self.fit(cells).transform(cells)

    def oov_report(self, cells: pd.Series) -> pd.DataFrame:
        """Determinants present in ``cells`` but not in the fitted vocabulary."""
        counts: dict[tuple[str, str], int] = {}
        for cell in cells:
            for kind, token in tokenise_amr_cell(cell):
                if kind in self.include_kinds and f"{kind}__{token}" not in self._index:
                    counts[(kind, token)] = counts.get((kind, token), 0) + 1
        return pd.DataFrame(
            [
                {"feature_kind": kind, "determinant": token, "n_isolates": n}
                for (kind, token), n in sorted(counts.items(), key=lambda r: -r[1])
            ],
            columns=["feature_kind", "determinant", "n_isolates"],
        )
