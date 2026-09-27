"""Build agrihgt_kaggle.ipynb from the percent-format source agrihgt_kaggle.py.

The evaluation script (../scripts/agrihgt_metrics.py) is embedded into the notebook so the
Kaggle run needs no extra files.

    python build_notebook.py
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "agrihgt_kaggle.py"
METRICS = HERE.parent / "scripts" / "agrihgt_metrics.py"
OUT = HERE / "agrihgt_kaggle.ipynb"


def cells_from_percent(text):
    cells = []
    for block in re.split(r"^# %%", text, flags=re.M)[1:]:
        header, _, body = block.partition("\n")
        body = body.strip("\n")
        if "[markdown]" in header:
            lines = [re.sub(r"^# ?", "", ln) for ln in body.splitlines()]
            cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": body})
    for c in cells:
        c["source"] = [ln + "\n" for ln in c["source"].split("\n")]
        c["source"][-1] = c["source"][-1].rstrip("\n")
    return cells


def main():
    metrics = METRICS.read_text()
    assert "'''" not in metrics, "metrics script must not contain triple single quotes"
    text = SRC.read_text().replace("<<AGRIHGT_METRICS>>", metrics)
    nb = {
        "cells": cells_from_percent(text),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"wrote {OUT} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
