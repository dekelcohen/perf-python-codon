# CPython baseline: tight loop calling common data libraries K times.
# Inputs cycle over a pool of P prebuilt values; attribute lookups hoisted.
# Usage: python libs_py.py <case> <K>      Output: RESULT,<case>,<K>,<seconds>,<checksum>
import sys, time, json, re
import numpy as np
import pandas as pd
import polars as pl
import orjson

P = 1000


def pool_floats():
    return [float(i) + 0.5 for i in range(P)]


def pool_ints():
    return [(i * 7919) % P for i in range(P)]


def pool_dates():
    return [f"2024-{1 + i % 12:02d}-{1 + i % 28:02d} {i % 24:02d}:{i % 60:02d}:{(i * 7) % 60:02d}" for i in range(P)]


def pool_json():
    return ['{"a": %d, "b": [1, 2, 3], "c": "x%d"}' % (i, i) for i in range(P)]


def pool_text():
    return [f"id {i} qty {i * 3 % 97} ref {i * 7919 % 100003} end" for i in range(P)]


def pool_vecs():
    return [[float((i * 31 + j) % 17) for j in range(8)] for i in range(P)]


def main():
    case, k = sys.argv[1], int(sys.argv[2])
    tot = 0
    if case == "np_sqrt":
        f, xs = np.sqrt, pool_floats()
        t0 = time.perf_counter()
        for i in range(k):
            tot += int(f(xs[i % P]) * 1000)
    elif case == "np_dot":
        f, vs = np.dot, [np.array(v) for v in pool_vecs()]
        t0 = time.perf_counter()
        for i in range(k):
            tot += int(f(vs[i % P], vs[(i + 1) % P]))
    elif case == "pd_timestamp":
        f, ds = pd.Timestamp, pool_dates()
        t0 = time.perf_counter()
        for i in range(k):
            tot += int(f(ds[i % P]).value // 1_000_000_000)
    elif case == "pl_getitem":
        s, idx = pl.Series("v", pool_ints()), pool_ints()
        t0 = time.perf_counter()
        for i in range(k):
            tot += int(s[idx[i % P]])
    elif case == "json_loads":
        f, js = json.loads, pool_json()
        t0 = time.perf_counter()
        for i in range(k):
            tot += int(f(js[i % P])["a"])
    elif case == "orjson_loads":
        f, js = orjson.loads, pool_json()
        t0 = time.perf_counter()
        for i in range(k):
            tot += int(f(js[i % P])["a"])
    elif case == "re_findall":
        f, ts = re.compile(r"\d+").findall, pool_text()
        t0 = time.perf_counter()
        for i in range(k):
            tot += len(f(ts[i % P]))
    else:
        sys.exit("unknown case " + case)
    dt = time.perf_counter() - t0
    print(f"RESULT,{case},{k},{dt:.6f},{tot}")


if __name__ == "__main__":
    main()
