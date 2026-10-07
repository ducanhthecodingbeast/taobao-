"""T3a — single-node join (Direction 1 path): label x SCL embedding x item metadata.

Why this is the hard part of the project
----------------------------------------
The brief is right that the difficulty is data engineering, not the neural network. The
dataset deliberately shards the supervised signal away from the features:

* labels live in ``raw/*_samples.parquet``      (``label_0`` one-hot + user + item)
* item content lives in ``raw/item_features.parquet``
* the multimodal signal lives in ``raw/scl_embedding_int8_p90.parquet`` (35.4 M x 128 int8)

Naive ``pandas.merge`` on 76 M x 35.4 M is an out-of-core hash join and OOMs immediately.

DuckDB-specific problems and the fixes used here
------------------------------------------------
P1. ``label_0`` is ``list<int8>`` length 2, not a scalar.
    -> ``label_0[2]`` (DuckDB lists are 1-based) yields the click indicator.
P2. DuckDB defaults ``memory_limit`` to 80 % of *total* RAM (156 GiB) which would let it
    consume the 41 GiB that is actually free, and the box is shared.
    -> ``PRAGMA memory_limit='24GB'`` plus an explicit spill directory.
P3. The root filesystem has only ~101 GB free but ``/tmp`` is a **79 GB tmpfs**.
    -> ``PRAGMA temp_directory='/tmp/tmm_duckdb'`` keeps spills in RAM instead of filling the disk.
P4. Joining all 76 M rows before sampling is wasteful; the join should see the sample.
    -> CTE with ``USING SAMPLE reservoir(...) REPEATABLE(seed)`` and let DuckDB push the
       sample below the join.
P5. ``preserve_insertion_order`` forces a single-threaded materialisation path.
    -> disable it; row order is irrelevant here and we sort explicitly if needed.

The output is a *compact* training table: 128 int8 + ~40 B per row, so 2 M rows is ~350 MB
instead of the 1.55 TB the HuggingFace viewer reports for the fully materialised dataset.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .config import PATHS, SEED


def connect(memory_limit: str = "24GB", threads: int = 24, temp_dir: str = "/tmp/tmm_duckdb"):
    import duckdb

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(f"PRAGMA temp_directory='{temp_dir}'")
    con.execute("PRAGMA preserve_insertion_order=false")
    return con


def _timed(con, sql: str) -> tuple[list, float, str]:
    t = time.time()
    plan = con.execute("EXPLAIN " + sql).fetchall()
    cur = con.execute(sql)
    rows = cur.fetchall()
    return rows, time.time() - t, plan[0][1] if plan else ""


def join_samples(rows: int = 2_000_000, kind: str = "train", out: Path | None = None,
                 con=None) -> dict:
    """Materialise ``rows`` joined training samples. Returns timing + invariant stats."""
    con = con or connect()
    out = out or (PATHS.data / f"{kind}_joined_{rows}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)

    samples = PATHS.raw_dir / f"{kind}_samples.parquet"
    sql = f"""
    COPY (
      WITH s AS (
        SELECT "129_1" AS user_id, "205" AS item_id, label_0[2]::TINYINT AS label
        FROM read_parquet('{samples}')
        USING SAMPLE reservoir({rows} ROWS) REPEATABLE({SEED})
      )
      SELECT s.user_id, s.item_id, s.label,
             e."205_c"                AS emb,
             i."206"                  AS cat,
             i."213"                  AS city,
             i."214"                  AS province
      FROM s
      JOIN read_parquet('{PATHS.raw_dir}/scl_embedding_int8_p90.parquet') e
        ON e."205" = s.item_id
      LEFT JOIN read_parquet('{PATHS.raw_dir}/item_features.parquet') i
        ON i."205" = s.item_id
    ) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
    """
    t = time.time()
    con.execute(sql)
    elapsed = time.time() - t

    stats = con.execute(f"""
        SELECT count(*) n, avg(label) pos_rate, count(DISTINCT user_id) users,
               count(DISTINCT item_id) items,
               sum(CASE WHEN cat IS NULL THEN 1 ELSE 0 END) missing_meta,
               avg(len(emb)) emb_dim
        FROM read_parquet('{out}')
    """).fetchone()
    size = out.stat().st_size
    return {
        "output": str(out), "rows": int(stats[0]), "pos_rate": float(stats[1]),
        "users": int(stats[2]), "items": int(stats[3]),
        "missing_item_metadata": int(stats[4]), "emb_dim": float(stats[5]),
        "seconds": round(elapsed, 2),
        "rows_per_second": int(stats[0] / max(elapsed, 1e-6)),
        "bytes": size, "gb": round(size / 2**30, 3),
    }


def verify_no_inner_join_loss(rows_sample: int = 200_000, con=None) -> dict:
    """Invariant P6: an INNER JOIN silently drops rows when coverage < 100 %.

    The README claims 100 % embedding coverage for *target* items. If that were wrong, the
    inner join above would quietly shrink the dataset and bias the label distribution - the
    classic silent failure. This checks it explicitly on a sample.
    """
    con = con or connect()
    q = f"""
    WITH s AS (
      SELECT "205" AS item_id, label_0[2]::TINYINT AS label
      FROM read_parquet('{PATHS.raw_dir}/train_samples.parquet')
      USING SAMPLE reservoir({rows_sample} ROWS) REPEATABLE({SEED})
    )
    SELECT count(*) AS sampled,
           sum(CASE WHEN e."205" IS NOT NULL THEN 1 ELSE 0 END) AS matched,
           avg(s.label) AS pos_rate_before,
           avg(CASE WHEN e."205" IS NOT NULL THEN s.label END) AS pos_rate_after
    FROM s LEFT JOIN read_parquet('{PATHS.raw_dir}/scl_embedding_int8_p90.parquet') e
      ON e."205" = s.item_id
    """
    r = con.execute(q).fetchone()
    return {
        "sampled": int(r[0]), "matched": int(r[1]),
        "join_retention": round(r[1] / max(r[0], 1), 6),
        "pos_rate_before": float(r[2]), "pos_rate_after": float(r[3]),
        "pos_rate_shift": abs(float(r[2]) - float(r[3])),
    }


def run(rows: int = 2_000_000, out: Path | None = None) -> dict:
    t0 = time.time()
    con = connect()
    res = {"engine": "duckdb", "rows_requested": rows}
    res["no_join_loss"] = verify_no_inner_join_loss(con=con)
    res["join"] = join_samples(rows=rows, out=out, con=con)
    res["total_seconds"] = round(time.time() - t0, 2)
    res["problems"] = [
        "P1 label_0 is list<int8>, needs label_0[2]",
        "P2 memory_limit default 80% of 156GiB > 41GiB actually free",
        "P3 root fs ~101GB free -> spill to 79GB tmpfs at /tmp",
        "P4 sample must be pushed below the join",
        "P5 preserve_insertion_order=False for parallel materialisation",
    ]
    PATHS.bench.mkdir(parents=True, exist_ok=True)
    (PATHS.bench / "join_duckdb.json").write_text(json.dumps(res, indent=2, default=str))
    con.close()
    return res


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run(), indent=2, default=str))
