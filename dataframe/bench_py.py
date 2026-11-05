# Python side of the numeric / dataframe benchmarks.
#   python bench_py.py <bench> <impl> <N>
# bench: numpy_vec | numpy_loop | groupby
# impl : numpy | numba (numpy_* only) | pandas | polars (groupby only)
# Output: RESULT,<bench>,<impl>,<N>,<seconds>,<checksum>
import sys
import time

import numpy as np

# One warmup call (lib init / caches), then ONE timed call per process.
# The runner repeats processes and keeps the best, same as for Codon.
WARMUP = True


def make_data(n):
    # Deterministic, identical to the Codon side (no RNG differences).
    i = np.arange(n, dtype=np.int64)
    x = (i * 0.6180339887498949) % 1.0
    k = (i * 2654435761) % 1000
    return x, k


# ---------------- numpy_vec: fused elementwise math + reduction ----------------
def vec_numpy(x, k):
    return float((np.sin(x) * np.exp(-x) + np.sqrt(x)).sum())


# ---------------- numpy_loop: sequential (non-vectorizable) loop ----------------
def loop_python(x, k):
    ema = 0.0
    hits = 0
    for i in range(x.shape[0]):
        ema = 0.7 * ema + 0.3 * x[i]
        if ema > 0.55:
            hits += 1
            ema *= 0.5
    return hits + ema


def _numba_loop():
    import numba

    @numba.njit(cache=False)
    def f(x):
        ema = 0.0
        hits = 0
        for i in range(x.shape[0]):
            ema = 0.7 * ema + 0.3 * x[i]
            if ema > 0.55:
                hits += 1
                ema *= 0.5
        return hits + ema

    return f


def _numba_vec():
    import numba

    @numba.njit(cache=False)
    def f(x):
        s = 0.0
        for i in range(x.shape[0]):
            v = x[i]
            s += np.sin(v) * np.exp(-v) + np.sqrt(v)
        return s

    return f


# ---------------- groupby: sum/mean/count/max per key ----------------
def gb_pandas(x, k):
    import pandas as pd

    df = pd.DataFrame({"k": k, "v": x})
    g = df.groupby("k", sort=True)["v"].agg(["sum", "mean", "count", "max"])
    return float(g["mean"].sum() + g["max"].sum() + g["count"].sum())


def gb_polars(x, k):
    import polars as pl

    df = pl.DataFrame({"k": k, "v": x})
    g = df.group_by("k").agg(
        pl.col("v").sum().alias("s"),
        pl.col("v").mean().alias("m"),
        pl.len().alias("c"),
        pl.col("v").max().alias("mx"),
    )
    return float(g["m"].sum() + g["mx"].sum() + g["c"].sum())


def pick(bench, impl):
    table = {
        ("numpy_vec", "numpy"): lambda: vec_numpy,
        ("numpy_vec", "numba"): lambda: _wrap_numba(_numba_vec()),
        ("numpy_loop", "numpy"): lambda: loop_python,
        ("numpy_loop", "numba"): lambda: _wrap_numba(_numba_loop()),
        ("groupby", "pandas"): lambda: gb_pandas,
        ("groupby", "polars"): lambda: gb_polars,
    }
    if (bench, impl) not in table:
        raise SystemExit(f"unsupported combo {bench}/{impl}")
    return table[(bench, impl)]()


def _wrap_numba(jf):
    jf(np.zeros(4))  # trigger JIT compile outside the timed region

    def run(x, k):
        return float(jf(x))

    return run


def main():
    bench, impl, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    x, k = make_data(n)
    fn = pick(bench, impl)
    if WARMUP:
        fn(x, k)
    t0 = time.perf_counter()
    chk = fn(x, k)
    dt = time.perf_counter() - t0
    print(f"RESULT,{bench},{impl},{n},{dt:.6f},{chk:.4f}")


if __name__ == "__main__":
    main()
