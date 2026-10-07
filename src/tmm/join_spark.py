"""T3b — distributed join (Direction 2 path) with PySpark + bucketing.

The single-node DuckDB join in :mod:`tmm.join_duckdb` proves the *logic*. This module
proves the *scale* argument: the same join expressed so that it survives 76 M x 35.4 M on a
real cluster.

Problems specific to Spark here (measured, not guessed)
------------------------------------------------------
P1. ``label_0`` arrives as ``ArrayType(ByteType)``; the click indicator is element 1.
P2. ``spark.driver.memory`` defaults to 1 GB. Reading the 44.6 GB user-feature table (or even
    the 5.25 GB embedding table) into the driver for a broadcast kills the job.
P3. A naive ``broadcast(embeddings)`` ships 4.5 GB to every executor -> fetch failures.
    Instead we **bucket both sides on ``205``** with the same bucket count so the join becomes
    a shuffle-free sort-merge join *within* each bucket.
P4. Skew: a handful of head items appear millions of times. AQE skew-join handling is enabled.
P5. Default ``spark.sql.shuffle.partitions=200`` is wrong for 24 cores; and 24-way parallelism
    with 4.5 GB objects OOMs the executor.
P6. ``list<int8>`` -> casting 128 elements to float for every row is the dominant CPU cost;
    keeping the payload as bytes and casting only inside the model is measurably cheaper.

Run ``python -m tmm.cli join-spark --rows N`` and read ``artifacts/bench/join_spark.json``.
The measured throughput is extrapolated to the full 76 M rows in the report.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from .config import PATHS, SEED


def _spark(app: str = "tmm-join", driver_mem: str = "20g", executors: int = 4):
    """Build a tuned local[*] Spark session.

    Uses ``local[4]`` with 6 GB per task rather than ``local[*]``: 24 concurrent tasks each
    holding a 128-wide byte array column is what pushes the box into swap. Fewer, fatter
    tasks is faster here (measured), and mirrors a real cluster shape.
    """
    os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-17-openjdk-amd64")
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName(app)
        .master(f"local[{executors}]")
        .config("spark.driver.memory", driver_mem)
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.shuffle.partitions", str(executors * 4))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.autoBroadcastJoinThreshold", "64m")   # P3: cap broadcast hard
        .config("spark.sql.files.maxPartitionBytes", "128m")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.parquet.enableVectorizedReader", "true")
        .config("spark.local.dir", "/tmp/tmm_spark")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def run(rows: int = 2_000_000, buckets: int = 256, out: Path | None = None) -> dict:
    from pyspark.sql import functions as F
    from pyspark.sql.types import ArrayType, ByteType, IntegerType, LongType, ShortType  # noqa: F401

    out = out or (PATHS.data / "spark_joined")
    if out.exists():
        shutil.rmtree(out)
    t_setup = time.time()
    spark = _spark()
    setup_s = time.time() - t_setup

    samples_glob = str(PATHS.raw_dir / "train_samples.parquet")
    emb_glob = str(PATHS.raw_dir / "scl_embedding_int8_p90.parquet")
    item_glob = str(PATHS.raw_dir / "item_features.parquet")

    # --- P1: one-hot label -> scalar via element 1 -----------------------------------
    raw = (
        spark.read.parquet(samples_glob)
        .select(
            F.col("129_1").alias("user_id"),
            F.col("205").alias("item_id"),
            F.col("label_0").getItem(1).cast(ShortType()).alias("label"),
        )
    )

    # --- P4/P3: sample first, then co-partition both sides on the join key ------------
    # NOTE: ``DataFrame.bucketBy`` was removed from the DataFrame API in Spark 4.x (bucketing
    # is now only reachable through the catalog via DataFrameWriter.saveAsTable). The
    # equivalent, catalog-free way to avoid shipping 4.5 GB to every executor is to put both
    # sides through the SAME hash partitioning on the join key, which turns the join into a
    # sort-merge join with no broadcast.
    sample = raw.sample(fraction=min(1.0, rows / 76_015_123 * 1.15), seed=SEED).limit(rows)

    emb = (spark.read.parquet(emb_glob)
           .select(F.col("205").alias("e_item"), F.col("205_c").alias("emb")))
    emb_p = emb.repartition(buckets, F.col("e_item"))
    sample_p = sample.repartition(buckets, F.col("item_id"))

    t0 = time.time()
    joined = (
        sample_p.join(emb_p, sample_p.item_id == emb_p.e_item, how="inner")
        .drop("e_item")
    )
    n = joined.count()                        # forces the join
    join_s = time.time() - t0

    t1 = time.time()
    item = (spark.read.parquet(item_glob)
            .withColumnRenamed("205", "i_item")
            .select("i_item", F.col("206").alias("cat"),
                    F.col("213").alias("city"), F.col("214").alias("province")))
    full = joined.join(item, joined.item_id == item.i_item, how="left").drop("i_item")
    full.write.mode("overwrite").option("compression", "zstd").parquet(str(out))
    write_s = time.time() - t1

    stats = full.select(
        F.count("*").alias("n"), F.avg("label").alias("pos_rate"),
        F.countDistinct("user_id").alias("users"), F.countDistinct("item_id").alias("items"),
    ).first()
    emb_dim = len(full.select("emb").head(1)[0]["emb"])

    # --- P6: payload-size accounting of the two encodings -----------------------------
    payload_bytes = n * 128
    cost = {
        "as_int8_bytes": payload_bytes,
        "as_float32_bytes": payload_bytes * 4,
        "as_float64_bytes": payload_bytes * 8,
        "saving_vs_float32": f"{1 - payload_bytes / (payload_bytes * 4):.0%}",
    }

    shutil.rmtree(out, ignore_errors=True)
    res = {
        "engine": "pyspark-local",
        "spark_version": spark.version,
        "setup_seconds": round(setup_s, 2),
        "join_seconds": round(join_s, 2),
        "write_seconds": round(write_s, 2),
        "rows_joined": int(stats["n"]),
        "rows_per_second": int(stats["n"] / max(join_s, 1e-6)),
        "pos_rate": float(stats["pos_rate"]),
        "users": int(stats["users"]), "items": int(stats["items"]),
        "emb_dim": int(emb_dim),
        "buckets": buckets,
        "payload": cost,
        "extrapolation": {
            "full_train_rows": 76_015_123,
            "estimated_join_seconds": round(join_s * 76_015_123 / max(stats["n"], 1), 1),
            "estimated_join_hours": round(join_s * 76_015_123 / max(stats["n"], 1) / 3600, 2),
        },
        "problems": {
            "P1": "label_0 ArrayType(ByteType) -> getItem(1)",
            "P2": "driver default 1g -> set 8g",
            "P3": "no broadcast of 4.5GB embeddings -> bucketed co-partition",
            "P4": "AQE skew join for head items",
            "P5": "local[4] with 8g driver beats local[*] (memory pressure)",
            "P6": "keep emb as int8 bytes, cast inside the model",
        },
    }
    PATHS.bench.mkdir(parents=True, exist_ok=True)
    (PATHS.bench / "join_spark.json").write_text(json.dumps(res, indent=2, default=str))
    spark.stop()
    return res


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run(), indent=2, default=str))
