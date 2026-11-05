#!/usr/bin/env python
"""Build Codon binaries, run Python-vs-Codon benchmark suites, write results/.

Usage (inside `conda activate codon_eval`):
    python run_benchmarks.py                       # all suites
    python run_benchmarks.py --suite text accel    # selected suites
    python run_benchmarks.py --quick               # 2 smallest sizes only (sanity)

Each suite writes results/<suite>.csv; results/RESULTS.md is regenerated from all CSVs present.
"""
import argparse
import csv
import glob
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
REPS = 3
TIMEOUT = 900
PY = "{py}"

# Per suite: working dir, Codon builds (src, exe, extra flags), and benches.
# bench -> (sizes, [(column label, argv template)]). First column = speedup baseline.
# argv templates: "{py}" -> this interpreter, "{n}" -> size.
_FW = ["fw_lists", "fw_flat", "fw_simd", "fw_par", "fw_par_simd", "fw_blocked", "fw_blocked_par", "fw_gpu"]
SUITES = {
    "algo": dict(
        title="1. Classical algorithms - identical source, CPython vs Codon",
        dir="algo",
        builds=[("algos.py", "algos_codon", [])],
        benches={
            b: (sizes, [("CPython", [PY, "algos.py", b, "{n}"]), ("Codon", ["./algos_codon", b, "{n}"])])
            for b, sizes in {
                "floyd_warshall": [100, 200, 400, 800],
                "n_queens": [8, 10, 12, 13, 14],
                "lcs": [1000, 2000, 5000, 10000],
            }.items()
        },
    ),
    "accel": dict(
        title="2. Accelerating the classical algorithms in Codon (SIMD / multithreading / GPU)",
        note="Baseline = Codon compiled from the identical pure-Python source (`fw_lists` / `nq_serial`). "
             "`generic:*` = same code built with `--disable-native` (SSE2 only, no AVX2).",
        dir="algo",
        builds=[("algos_accel.codon", "algos_accel", ["--libdevice={libdevice}"]),
                ("algos_accel.codon", "algos_accel_generic", ["--disable-native", "--libdevice={libdevice}"])],
        benches={
            "floyd_warshall": ([800, 1600, 3200],
                               [(v, ["./algos_accel", v, "{n}"]) for v in _FW]
                               + [("generic:" + v, ["./algos_accel_generic", v, "{n}"]) for v in ("fw_simd", "fw_blocked")]),
            "n_queens": ([13, 14, 15, 16], [(v, ["./algos_accel", v, "{n}"]) for v in ("nq_serial", "nq_par")]),
        },
    ),
    "numba": dict(
        title="2b. Codon vs Numba on the classical algorithms (CPU serial / multithreaded / CUDA GPU)",
        note="`same` = the unchanged `algos.py` functions: compiled by Codon vs `@njit` (lists, recursion). "
             "Other columns: equivalent optimized variants (`algos_accel.codon` vs `algos_numba.py`). "
             "Numba JIT compile time is excluded (warm-up on tiny N); Numba threading layer = workqueue. "
             "GPU timings include host<->device copies.",
        dir="algo",
        builds=[("algos.py", "algos_codon", []),
                ("algos_accel.codon", "algos_accel", ["--libdevice={libdevice}"])],
        benches={
            **{f"same:{b}": (sizes, [("codon", ["./algos_codon", b, "{n}"]),
                                     ("numba", [PY, "algos_numba.py", f"same:{b}", "{n}"])])
               for b, sizes in {"floyd_warshall": [400, 800, 1600], "n_queens": [13, 14, 15],
                                "lcs": [5000, 10000, 20000]}.items()},
            "floyd_warshall": ([800, 1600, 3200],
                               [(f"{rt}:{v}", ["./algos_accel", v, "{n}"] if rt == "codon"
                                 else [PY, "algos_numba.py", v, "{n}"])
                                for v in ("fw_flat", "fw_par", "fw_blocked", "fw_blocked_par", "fw_gpu")
                                for rt in ("codon", "numba")]),
            "n_queens": ([14, 15, 16],
                         [(f"{rt}:{v}", ["./algos_accel", v, "{n}"] if rt == "codon"
                           else [PY, "algos_numba.py", v, "{n}"])
                          for v in ("nq_serial", "nq_par", "nq_gpu") for rt in ("codon", "numba")]),
        },
    ),
    "dataframe": dict(
        title="3. NumPy / DataFrame workloads",
        dir="dataframe",
        builds=[("bench_codon.codon", "bench_codon", [])],
        benches={
            b: ([100_000, 1_000_000, 10_000_000],
                [(f"{rt}:{impl}", ([PY, "bench_py.py"] if rt == "python" else ["./bench_codon"]) + [b, impl, "{n}"])
                 for rt, impl in impls])
            for b, impls in {
                "numpy_vec": [("python", "numpy"), ("python", "numba"), ("codon", "native"), ("codon", "native_par")],
                "numpy_loop": [("python", "numpy"), ("python", "numba"), ("codon", "native")],
                "groupby": [("python", "pandas"), ("python", "polars"), ("codon", "native"), ("codon", "polars")],
            }.items()
        },
    ),
    "text": dict(
        title="4. Text / strings / regex",
        note="`python:pure` and `codon:pure` run the identical source (`text_bench.py`). Codon `re` is RE2-based. "
             "Other columns are optimized Python libraries. N = words (text) or string length (levenshtein).",
        dir="text",
        builds=[("text_bench.py", "text_codon", [])],
        benches={
            b: (sizes,
                [("python:pure", [PY, "text_bench.py", b, "{n}"]), ("codon:pure", ["./text_codon", b, "{n}"])]
                + [(f"python:{impl}", [PY, "text_py_fast.py", b, impl, "{n}"]) for impl in fast])
            for b, sizes, fast in [
                ("word_freq", [100_000, 1_000_000, 3_000_000], ["counter", "polars"]),
                ("regex_findall", [100_000, 1_000_000, 3_000_000], ["re2", "polars"]),
                ("regex_sub", [100_000, 1_000_000, 3_000_000], ["re2", "polars"]),
                ("str_methods", [100_000, 1_000_000, 3_000_000], []),
                ("levenshtein", [1000, 3000, 10000], ["rapidfuzz"]),
            ]
        },
    ),
    "interop": dict(
        title="5. Calling rapidfuzz from Codon in a tight loop: Python interop vs direct C/C++",
        note="M string pairs (len 32, ~30% edits), sum of Levenshtein distances. `codon:py_*` use "
             "`from python import rapidfuzz` (CPython C-API: str->PyUnicode copies, arg tuple, PyObject_Call, "
             "result unboxing). `codon:c_shim*` call rapidfuzz-cpp through a 5-line `extern \"C\"` shim via "
             "`from C import`. `codon:py_noop` calls Python `len()` = pure per-call interop overhead.",
        dir="interop",
        prebuild=[["g++", "-O3", "-march=native", "-std=c++17", "-shared", "-fPIC", "-I{prefix}/include",
                   "-o", "librf_shim.so", "rf_shim.cpp"]],
        builds=[("interop_bench.codon", "interop_codon",
                 ["-l", "rf_shim", "--linker-flags=-L. -Wl,-rpath,$ORIGIN"])],
        benches={
            "levenshtein_pairs": ([100_000, 1_000_000, 3_000_000],
                                  [(f"python:{i}", [PY, "interop_py.py", i, "{n}"]) for i in ("loop", "cpdist", "cpdist_mt")]
                                  + [(f"codon:{i}", ["./interop_codon", i, "{n}"])
                                     for i in ("py_loop", "py_preconv", "py_noop", "py_cpdist",
                                               "c_shim", "c_shim_par", "native_dp")]),
        },
    ),
    "interop_libs": dict(
        title="6. Calling common Python data libraries in a tight loop: CPython vs Codon `from python import`",
        note="N = number of calls; inputs cycle over a pool of 1000 prebuilt values; attribute lookups hoisted. "
             "`codon:conv` passes native Codon values (converted per call), `codon:pre` passes pre-converted "
             "pyobjs, `codon:conv_bigheap` = conv with 3M extra live Codon strings on the GC heap.",
        dir="interop",
        builds=[("libs_codon.codon", "libs_codon", [])],
        benches={
            case: ([100_000, 1_000_000],
                   [("python:loop", [PY, "libs_py.py", case, "{n}"])]
                   + [(f"codon:{m}", ["./libs_codon", case, m, "{n}"]) for m in ("conv", "pre", "conv_bigheap")])
            for case in ("np_sqrt", "np_dot", "pd_timestamp", "pl_getitem", "json_loads", "orjson_loads", "re_findall")
        },
    ),
    "geo": dict(
        title="7. Geo: are 3M points inside any of 200 polygons? shapely (CPython) vs GEOS C API / native Codon",
        note="200 star-shaped polygons (16-63 vertices) on a 1000x1000 map; ~21% of points are inside. "
             "`python:shapely_*` = shapely 2.2 / GEOS 3.14 from CPython (`loop` = per-point Python loop with bbox "
             "prefilter + `contains_xy` on prepared polygons, `union_loop` = one call per point on the prepared union, "
             "`vec` = one vectorized call, `strtree` = STRtree query). `python:pure`/`codon:pure` = identical "
             "ray-casting source (`geo_bench.py`). `codon:geos*` call libgeos_c directly via `from C import`. "
             "Grid = 100x100 cells: per cell 'fully inside a polygon' flag or CSR list of candidate polygons "
             "(grid build is inside the timed region). GPU times include host->device copies.",
        dir="geo",
        builds=[("geo_bench.py", "geo_same", []),
                ("geo_codon.codon", "geo_codon",
                 ["-l", "geos_c", "--linker-flags=-L{prefix}/lib -Wl,-rpath,{prefix}/lib", "--libdevice={libdevice}"])],
        benches={
            "pip_naive": ([100_000, 1_000_000, 3_000_000],
                          [("python:shapely_loop", [PY, "geo_shapely.py", "loop", "{n}"]),
                           ("python:pure", [PY, "geo_bench.py", "naive", "{n}"]),
                           ("python:shapely_union_loop", [PY, "geo_shapely.py", "union_loop", "{n}"]),
                           ("python:shapely_vec", [PY, "geo_shapely.py", "vec", "{n}"]),
                           ("python:shapely_strtree", [PY, "geo_shapely.py", "strtree", "{n}"]),
                           ("codon:pure", ["./geo_same", "naive", "{n}"])]
                          + [(f"codon:{v}", ["./geo_codon", v, "{n}"]) for v in ("geos", "geos_par", "pure_par", "gpu")]),
            "pip_grid": ([100_000, 1_000_000, 3_000_000],
                         [("python:shapely_grid", [PY, "geo_shapely.py", "grid_loop", "{n}"]),
                          ("python:pure_grid", [PY, "geo_bench.py", "grid", "{n}"]),
                          ("codon:pure_grid", ["./geo_same", "grid", "{n}"])]
                         + [(f"codon:{v}", ["./geo_codon", v, "{n}"]) for v in ("geos_grid", "grid_par", "grid_gpu")]),
        },
    ),
}


def sh(cmd, cwd):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=TIMEOUT)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{p.stdout}\n{p.stderr}")
    return p.stdout, time.perf_counter() - t0


def build(suite, cfg):
    if not shutil.which("codon"):
        sys.exit("codon not on PATH - run `conda activate codon_eval` first")
    libdevice = os.environ.get("CODON_LIBDEVICE", "")
    times = {}
    for cmd in cfg.get("prebuild", []):
        sh([c.format(prefix=sys.prefix) for c in cmd], os.path.join(HERE, cfg["dir"]))
    for src, exe, flags in cfg["builds"]:
        flags = [f.format(libdevice=libdevice, prefix=sys.prefix) for f in flags]
        _, dt = sh(["codon", "build", "-release", *flags, "-o", exe, src], os.path.join(HERE, cfg["dir"]))
        times[exe] = dt
        print(f"[{suite}] built {exe} in {dt:.1f}s")
    return times


def best_of(argv, cwd):
    best, chk = float("inf"), None
    for _ in range(REPS):
        out, _ = sh(argv, cwd)
        line = next(l for l in out.splitlines() if l.startswith("RESULT,"))
        parts = line.split(",")
        best = min(best, float(parts[-2]))
        chk = parts[-1]
    return best, chk


def run_suite(suite, quick):
    cfg = SUITES[suite]
    builds = build(suite, cfg)
    cwd = os.path.join(HERE, cfg["dir"])
    rows = []
    for bench, (sizes, cols) in cfg["benches"].items():
        for n in sizes[:2] if quick else sizes:
            for label, tmpl in cols:
                argv = [a.replace("{py}", sys.executable).replace("{n}", str(n)) for a in tmpl]
                sec, chk = best_of(argv, cwd)
                rows.append((suite, bench, label, n, sec, chk))
                print(f"[{suite}] {bench:15s} n={n:<10} {label:22s} {sec:10.4f}s  chk={chk}")
    with open(os.path.join(OUT, f"{suite}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["suite", "bench", "impl", "n", "seconds", "checksum"])
        w.writerows(rows)
    with open(os.path.join(OUT, f"{suite}.build.csv"), "w", newline="") as f:
        csv.writer(f).writerows(builds.items())


def fmt(s):
    if s < 1e-3:
        return f"{s * 1e6:.0f} us"
    return f"{s * 1e3:.2f} ms" if s < 1 else f"{s:.2f} s"


def write_report():
    import numba
    import numpy
    import pandas
    import polars

    rows, builds = [], {}
    for path in sorted(glob.glob(os.path.join(OUT, "*.csv"))):
        with open(path) as f:
            if path.endswith(".build.csv"):
                builds.update({k: float(v) for k, v in csv.reader(f)})
            else:
                rows += [(r["suite"], r["bench"], r["impl"], int(r["n"]), float(r["seconds"]), r["checksum"])
                         for r in csv.DictReader(f)]

    chks = defaultdict(set)
    for r in rows:
        if "noop" not in r[2]:  # overhead probes compute something else on purpose
            chks[(r[1], r[3])].add(round(float(r[5]), 2))
    mismatches = {f"{b}@{n}": sorted(v) for (b, n), v in chks.items() if len(v) > 1}

    codon_ver = subprocess.run(["codon", "--version"], capture_output=True, text=True).stdout.strip()
    cpu = platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    except Exception:
        pass
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip() or "none"
    L = ["# Codon vs Python benchmark results", "",
         f"- CPU: {cpu} ({os.cpu_count()} threads); GPU: {gpu}",
         f"- Python {platform.python_version()}, Codon {codon_ver}, NumPy {numpy.__version__}, "
         f"numba {numba.__version__}, pandas {pandas.__version__}, polars {polars.__version__}",
         f"- Time = best of {REPS} separate processes; excludes process start, imports and data generation.",
         "- Codon compile time (`codon build -release`): " + ", ".join(f"`{k}` {v:.1f}s" for k, v in builds.items()),
         f"- Checksums identical across all implementations: **{'NO - ' + str(mismatches) if mismatches else 'yes'}**",
         "- Cells: time (speedup vs first column).", ""]

    t = {(r[0], r[1], r[2], r[3]): r[4] for r in rows}
    for suite, cfg in SUITES.items():
        if not any(r[0] == suite for r in rows):
            continue
        L += [f"## {cfg['title']}", ""]
        if cfg.get("note"):
            L += [cfg["note"], ""]
        for bench, (sizes, cols) in cfg["benches"].items():
            labels = [c[0] for c in cols]
            L += [f"### {bench}", "", "| N | " + " | ".join(labels) + " |", "|---:|" + "---:|" * len(labels)]
            for n in sizes:
                base = t.get((suite, bench, labels[0], n))
                if base is None:
                    continue
                cells = []
                for lab in labels:
                    s = t.get((suite, bench, lab, n))
                    cells.append("-" if s is None else f"{fmt(s)} ({base / s:.1f}x)")
                L.append(f"| {n:,} | " + " | ".join(cells) + " |")
            L.append("")
    with open(os.path.join(OUT, "RESULTS.md"), "w") as f:
        f.write("\n".join(L))
    return mismatches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", nargs="+", choices=list(SUITES), default=list(SUITES))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for s in args.suite:
        run_suite(s, args.quick)
    mismatches = write_report()
    print(f"\nwrote {OUT}/RESULTS.md")
    if mismatches:
        sys.exit(f"CHECKSUM MISMATCH: {mismatches}")


if __name__ == "__main__":
    main()
