"""Central configuration: paths, budgets, reproducibility knobs.

Every path is overridable by environment variable so the same code runs on a laptop
subset (Direction 1) and against the full 139 GB tree (Direction 2).
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("TMM_HOME", Path(__file__).resolve().parents[2]))
DATA_ROOT = Path(os.environ.get("TMM_DATA", "/home/aiface/Taobao-MM"))
ARTIFACTS = Path(os.environ.get("TMM_ARTIFACTS", PROJECT_ROOT / "artifacts"))

SEED = 42

#: Column contract of the dataset, verified against ``metadata.json``.
SAMPLE_COLS = ["label_0", "129_1", "205"]
JOINED_COLS = [
    "label_0", "129_1", "130_1", "130_2", "130_3", "130_4", "130_5",
    "150_2_180", "151_2_180", "205", "206", "213", "214",
]
HIST_ITEM_COL = "150_2_180"
HIST_CAT_COL = "151_2_180"


@dataclasses.dataclass
class Paths:
    data_root: Path = DATA_ROOT
    artifacts: Path = ARTIFACTS

    @property
    def train_dir(self) -> Path: return self.data_root / "train"

    @property
    def test_dir(self) -> Path: return self.data_root / "test"

    @property
    def raw_dir(self) -> Path: return self.data_root / "raw"

    @property
    def feature_map(self) -> Path: return self.data_root / "feature_map"

    @property
    def emb_keys(self) -> Path: return self.feature_map / "scl_emb_int8_p90_keys.npy"

    @property
    def emb_values(self) -> Path: return self.feature_map / "scl_emb_int8_p90_values.npy"

    @property
    def stats(self) -> Path: return self.artifacts / "stats"

    @property
    def figures(self) -> Path: return self.artifacts / "figures"

    @property
    def models(self) -> Path: return self.artifacts / "models"

    @property
    def bench(self) -> Path: return self.artifacts / "bench"

    @property
    def data(self) -> Path: return self.artifacts / "data"

    @property
    def reports(self) -> Path: return self.artifacts / "reports"

    def ensure(self) -> "Paths":
        for d in (self.stats, self.figures, self.models, self.bench, self.data, self.reports):
            d.mkdir(parents=True, exist_ok=True)
        return self


@dataclasses.dataclass
class Budget:
    """Hard caps that keep every stage inside the 41 GiB free RAM / 101 GB free disk."""

    #: rows sampled per stage when a full pass is impossible
    profile_rows: int = 200_000
    #: how many training samples to materialise (Direction 2 "scaled" run)
    train_rows: int = 2_000_000
    test_rows: int = 400_000
    #: number of catalogue items in the Direction 1 demo index
    demo_items: int = 10_000
    #: candidate/negative budget
    n_negatives: int = 500
    #: history length used by the ranker (MUSE searches inside this)
    seq_len: int = 1000
    #: MUSE sub-sequence size k
    muse_k: int = 50
    batch_size: int = 512
    epochs: int = 3
    lr: float = 3e-3
    emb_dim: int = 128


PATHS = Paths()
BUDGET = Budget()


def device(requested: str | None = None) -> str:
    """Resolve the compute device, honouring ``TMM_DEVICE`` and falling back to CPU."""
    import torch

    want = requested or os.environ.get("TMM_DEVICE", "auto")
    if want == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if want.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable. If running under a sandbox, the nvidia device "
            "nodes may be hidden; use escalated permissions for GPU stages."
        )
    return want
