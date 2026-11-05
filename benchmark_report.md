# Codon vs CPython — benchmark report

**Question:** how much does [Codon](https://github.com/exaloop/codon) 0.20.6 (AOT, LLVM-based Python-subset compiler)
speed up classical heavy algorithms, numeric/dataframe and text workloads; what extra acceleration
(SIMD/AVX, multithreading, GPU) it offers; and what it costs when Codon code calls Python libraries.

## When to use Codon — and when not

All numbers are from this report (i7-10700 8C/16T, RTX 3070). "vs CPython" = same task in plain CPython.

| ✅ Use Codon when… | Evidence (this report) | Typical gain |
|---|---|---|
| CPU-bound **pure-Python loops** (graphs, DP, backtracking, simulations) with no/low library use | Floyd–Warshall N=800 15.3 s → 0.47 s; N-Queens 14 7.4 s → 0.22 s; LCS 10k 7.6 s → 0.33 s | **15–35×** vs CPython, same source |
| You need **real multithreading** (no GIL) or **GPU** with Python syntax | `@par` N-Queens 8.7×, cache-blocked FW 23×; `@gpu.kernel` FW 44× (vs Codon serial) | 100–1000×+ vs CPython |
| A hot loop calls a **C/C++ library per element** | `from C import`: rapidfuzz-cpp 1.8× faster than CPython→rapidfuzz (10.8× with `@par`); GEOS C API 14.5× faster than CPython→shapely loop (132× with `@par`) | 2–130× |
| Custom **geometric/spatial kernels** (point-in-polygon, grids, indexes) | 3M points × 200 polygons: shapely loop 17.8 s, shapely vectorized 1.58 s → Codon grid+`@par` **20 ms**, GPU 21 ms | 77× vs best Python, 870× vs loop |
| Code is **not NumPy-shaped** (lists, dicts, strings, classes, recursion) and must be fast | Numba can't compile most of this; Codon handles it; N-Queens recursion Codon 1.1–1.4× faster than Numba | — |
| You want a **standalone native binary** | `codon build` → executable, no Python needed (unless `from python import`) | — |

| ❌ Don't use Codon (or expect nothing) when… | Evidence | Result |
|---|---|---|
| Code is already **vectorized NumPy / Polars** | `numpy_vec` 10M: NumPy 147 ms vs Codon 134 ms | ≈1× (only `@par` adds 5×) |
| A tight loop calls **Python libraries per element** (`from python import`) | numpy/pandas/polars/json/orjson/re/rapidfuzz per call | **2–5.5× SLOWER** than CPython; 10–40× slower with a large Codon heap (superlinear) |
| Work is **string/dict/regex-heavy** | word_freq 2.5× (0.7× at 100k), `re.sub` 1.5×, Polars `.str` as fast from Python | 0.4–6×, often not worth a rewrite |
| Numeric array loops **inside an existing Python/NumPy app** | Numba = same speed on identical source, 1.2–2.6× faster than Codon on optimized Floyd–Warshall CPU variants, GPU identical; no separate build | prefer **Numba** |
| A **better algorithm/library** exists | rapidfuzz bit-parallel Levenshtein ~470× faster than Codon-compiled DP; grid index 50× over naive scan | algorithm > compiler |
| You rely on dynamic Python / full ecosystem | Codon is a static subset; builds take 5–12 s per binary; hit 2 Codon `-release` miscompiles/GC bugs here (§9) | friction |

## TL;DR

| Area | Result |
|---|---|
| Pure-Python algorithms (identical source) | Codon **15–35× faster** than CPython |
| + Codon `@par` / cache blocking / AVX2 / GPU | further **9–44×** on top (Floyd–Warshall N=3200: 33.3 s → 1.42 s CPU, 0.75 s GPU; ~1000×+ vs CPython) |
| Vectorized NumPy | Codon ≈ NumPy (both run compiled kernels); `@par` loop 5.4× at 10M |
| Python loop over arrays | Codon ~52× vs CPython (numba ~94×) |
| Group-by 10M rows | native Codon 69 ms < Polars-from-Python 94 ms < Polars-from-Codon 106 ms < pandas 264 ms |
| Text / regex / dict counting | Codon only **1.5–6.5×** (and slower at small N); Polars `.str` comparable |
| Codon → Python library calls in a loop | Codon is **2–5.5× SLOWER** than CPython per call, and **10–40× slower** when Codon holds a big heap |
| Codon → C/C++ (`from C import`) | 1.8× faster than CPython→rapidfuzz serial, **10.8×** with `@par` |
| Codon vs **Numba** (classical algos) | identical source: parity (±10% at large N). Optimized Floyd–Warshall CPU: Numba 1.2–2.6× faster; N-Queens: Codon 1.1–1.4× faster; CUDA: identical. **Numba is not strictly better** |
| Geo: 3M points in any of 200 polygons | CPython+shapely loop 17.8 s · shapely vectorized 1.58 s · Codon→GEOS C API 1.23 s (134 ms `@par`) · pure Codon 0.99 s (141 ms `@par`, 55 ms GPU) · **grid index: Python 3.55 s → Codon 183 ms → `@par` 20 ms** |

**Rule of thumb:** Codon is great for hot loops written *in Codon* (or calling C directly); it is a poor host for
tight loops of *Python-library* calls. Call Python libs from Codon only for coarse-grained, vectorized work
(one call processing a whole array/frame).

## Environment & methodology

- CPU: Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz (16 threads); GPU: NVIDIA GeForce RTX 3070
- Python 3.12.16, Codon 0.20.6, NumPy 2.4.6, numba 0.68.1, pandas 3.1.1, polars 2.1.0
- Time = best of 3 separate processes; excludes process start, imports and data generation.
- Codon compile time (`codon build -release`): `algos_accel` 11.3s, `algos_accel_generic` 10.1s, `algos_codon` 5.4s, `bench_codon` 13.0s, `geo_same` 5.6s, `geo_codon` 11.7s, `interop_codon` 8.1s, `libs_codon` 8.2s, `text_codon` 10.9s
- Checksums identical across all implementations: **yes**
- Cells: time (speedup vs first column).

- Conda env `codon_eval` (Python 3.12, numpy, numba, pandas, polars, pyarrow, google-re2, rapidfuzz(+cpp), orjson,
  shapely 2.2 / GEOS 3.14, numba-cuda, cuda-nvvm-tools 12.2). Codon installed inside the env; activation hooks set `PATH`, `CODON_PYTHON`, `LIBRARY_PATH`,
  `CODON_LIBDEVICE`. All builds: `codon build -release` (host-native target, i.e. AVX2 on this CPU).
- Every (bench, impl, N) runs in a fresh process, best of 3. Data are generated deterministically (LCG) and every
  implementation prints a checksum; the runner verifies all checksums agree.
- Reproduce: `conda activate codon_eval && cd research && python run_benchmarks.py [--suite ...] [--quick]`.

## Detailed results

## 1. Classical algorithms - identical source, CPython vs Codon

### floyd_warshall

| N | CPython | Codon |
|---:|---:|---:|
| 100 | 37.37 ms (1.0x) | 1.70 ms (22.0x) |
| 200 | 256.64 ms (1.0x) | 23.10 ms (11.1x) |
| 400 | 1.95 s (1.0x) | 88.48 ms (22.0x) |
| 800 | 15.34 s (1.0x) | 466.94 ms (32.8x) |

### n_queens

| N | CPython | Codon |
|---:|---:|---:|
| 8 | 388 us (1.0x) | 20 us (19.4x) |
| 10 | 9.37 ms (1.0x) | 329 us (28.5x) |
| 12 | 229.99 ms (1.0x) | 11.99 ms (19.2x) |
| 13 | 1.26 s (1.0x) | 84.84 ms (14.8x) |
| 14 | 7.36 s (1.0x) | 218.94 ms (33.6x) |

### lcs

| N | CPython | Codon |
|---:|---:|---:|
| 1,000 | 63.61 ms (1.0x) | 2.60 ms (24.5x) |
| 2,000 | 277.70 ms (1.0x) | 14.39 ms (19.3x) |
| 5,000 | 1.81 s (1.0x) | 128.27 ms (14.1x) |
| 10,000 | 7.62 s (1.0x) | 328.31 ms (23.2x) |

## 2. Accelerating the classical algorithms in Codon (SIMD / multithreading / GPU)

Baseline = Codon compiled from the identical pure-Python source (`fw_lists` / `nq_serial`). `generic:*` = same code built with `--disable-native` (SSE2 only, no AVX2).

### floyd_warshall

| N | fw_lists | fw_flat | fw_simd | fw_par | fw_par_simd | fw_blocked | fw_blocked_par | fw_gpu | generic:fw_simd | generic:fw_blocked |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 800 | 420.28 ms (1.0x) | 207.89 ms (2.0x) | 153.60 ms (2.7x) | 63.55 ms (6.6x) | 74.00 ms (5.7x) | 111.62 ms (3.8x) | 35.67 ms (11.8x) | 14.30 ms (29.4x) | 333.93 ms (1.3x) | 333.34 ms (1.3x) |
| 1,600 | 4.20 s (1.0x) | 2.59 s (1.6x) | 2.10 s (2.0x) | 1.04 s (4.0x) | 1.06 s (4.0x) | 878.74 ms (4.8x) | 190.11 ms (22.1x) | 99.42 ms (42.3x) | 3.54 s (1.2x) | 2.73 s (1.5x) |
| 3,200 | 33.29 s (1.0x) | 23.90 s (1.4x) | 19.45 s (1.7x) | 18.38 s (1.8x) | 19.66 s (1.7x) | 7.30 s (4.6x) | 1.42 s (23.4x) | 750.19 ms (44.4x) | 30.93 s (1.1x) | 22.15 s (1.5x) |

### n_queens

| N | nq_serial | nq_par |
|---:|---:|---:|
| 13 | 31.33 ms (1.0x) | 10.88 ms (2.9x) |
| 14 | 196.27 ms (1.0x) | 27.69 ms (7.1x) |
| 15 | 1.13 s (1.0x) | 133.26 ms (8.5x) |
| 16 | 7.49 s (1.0x) | 863.10 ms (8.7x) |

## 2b. Codon vs Numba on the classical algorithms (CPU serial / multithreaded / CUDA GPU)

`same` = the unchanged `algos.py` functions: compiled by Codon vs `@njit` (lists, recursion). Other columns: equivalent optimized variants (`algos_accel.codon` vs `algos_numba.py`). Numba JIT compile time is excluded (warm-up on tiny N); Numba threading layer = workqueue. GPU timings include host<->device copies.

### same:floyd_warshall

| N | codon | numba |
|---:|---:|---:|
| 400 | 99.26 ms (1.0x) | 57.15 ms (1.7x) |
| 800 | 450.53 ms (1.0x) | 417.93 ms (1.1x) |
| 1,600 | 4.56 s (1.0x) | 4.09 s (1.1x) |

### same:n_queens

| N | codon | numba |
|---:|---:|---:|
| 13 | 43.81 ms (1.0x) | 33.70 ms (1.3x) |
| 14 | 208.28 ms (1.0x) | 198.38 ms (1.0x) |
| 15 | 1.18 s (1.0x) | 1.24 s (0.9x) |

### same:lcs

| N | codon | numba |
|---:|---:|---:|
| 5,000 | 113.48 ms (1.0x) | 74.06 ms (1.5x) |
| 10,000 | 329.12 ms (1.0x) | 313.36 ms (1.1x) |
| 20,000 | 1.26 s (1.0x) | 1.26 s (1.0x) |

### floyd_warshall

| N | codon:fw_flat | numba:fw_flat | codon:fw_par | numba:fw_par | codon:fw_blocked | numba:fw_blocked | codon:fw_blocked_par | numba:fw_blocked_par | codon:fw_gpu | numba:fw_gpu |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 800 | 213.82 ms (1.0x) | 83.84 ms (2.6x) | 62.77 ms (3.4x) | 50.44 ms (4.2x) | 117.97 ms (1.8x) | 84.00 ms (2.5x) | 29.89 ms (7.2x) | 17.03 ms (12.6x) | 14.28 ms (15.0x) | 20.96 ms (10.2x) |
| 1,600 | 2.66 s (1.0x) | 1.41 s (1.9x) | 1.01 s (2.6x) | 488.37 ms (5.4x) | 899.24 ms (3.0x) | 698.07 ms (3.8x) | 180.38 ms (14.7x) | 131.16 ms (20.3x) | 99.24 ms (26.8x) | 99.89 ms (26.6x) |
| 3,200 | 23.73 s (1.0x) | 12.99 s (1.8x) | 18.63 s (1.3x) | 7.27 s (3.3x) | 7.18 s (3.3x) | 5.84 s (4.1x) | 1.36 s (17.5x) | 1.16 s (20.4x) | 750.05 ms (31.6x) | 757.92 ms (31.3x) |

### n_queens

| N | codon:nq_serial | numba:nq_serial | codon:nq_par | numba:nq_par | codon:nq_gpu | numba:nq_gpu |
|---:|---:|---:|---:|---:|---:|---:|
| 14 | 185.08 ms (1.0x) | 201.61 ms (0.9x) | 27.10 ms (6.8x) | 24.73 ms (7.5x) | 10.95 ms (16.9x) | 11.36 ms (16.3x) |
| 15 | 1.14 s (1.0x) | 1.26 s (0.9x) | 138.92 ms (8.2x) | 188.81 ms (6.0x) | 75.38 ms (15.1x) | 75.46 ms (15.1x) |
| 16 | 7.48 s (1.0x) | 8.25 s (0.9x) | 849.82 ms (8.8x) | 972.40 ms (7.7x) | 686.20 ms (10.9x) | 689.49 ms (10.8x) |

## 3. NumPy / DataFrame workloads

### numpy_vec

| N | python:numpy | python:numba | codon:native | codon:native_par |
|---:|---:|---:|---:|---:|
| 100,000 | 1.50 ms (1.0x) | 996 us (1.5x) | 3.53 ms (0.4x) | 2.17 ms (0.7x) |
| 1,000,000 | 13.40 ms (1.0x) | 10.05 ms (1.3x) | 17.17 ms (0.8x) | 4.59 ms (2.9x) |
| 10,000,000 | 147.15 ms (1.0x) | 100.72 ms (1.5x) | 133.84 ms (1.1x) | 27.33 ms (5.4x) |

### numpy_loop

| N | python:numpy | python:numba | codon:native |
|---:|---:|---:|---:|
| 100,000 | 18.50 ms (1.0x) | 193 us (95.9x) | 345 us (53.6x) |
| 1,000,000 | 185.59 ms (1.0x) | 2.00 ms (93.0x) | 3.62 ms (51.2x) |
| 10,000,000 | 1.91 s (1.0x) | 20.36 ms (94.0x) | 36.48 ms (52.5x) |

### groupby

| N | python:pandas | python:polars | codon:native | codon:polars |
|---:|---:|---:|---:|---:|
| 100,000 | 4.10 ms (1.0x) | 831 us (4.9x) | 797 us (5.1x) | 1.55 ms (2.6x) |
| 1,000,000 | 27.14 ms (1.0x) | 3.72 ms (7.3x) | 7.15 ms (3.8x) | 6.71 ms (4.0x) |
| 10,000,000 | 263.94 ms (1.0x) | 94.40 ms (2.8x) | 69.09 ms (3.8x) | 106.43 ms (2.5x) |

## 4. Text / strings / regex

`python:pure` and `codon:pure` run the identical source (`text_bench.py`). Codon `re` is RE2-based. Other columns are optimized Python libraries. N = words (text) or string length (levenshtein).

### word_freq

| N | python:pure | codon:pure | python:counter | python:polars |
|---:|---:|---:|---:|---:|
| 100,000 | 30.69 ms (1.0x) | 41.67 ms (0.7x) | 27.76 ms (1.1x) | 12.32 ms (2.5x) |
| 1,000,000 | 280.73 ms (1.0x) | 110.76 ms (2.5x) | 266.52 ms (1.1x) | 132.74 ms (2.1x) |
| 3,000,000 | 804.77 ms (1.0x) | 323.78 ms (2.5x) | 764.53 ms (1.1x) | 401.47 ms (2.0x) |

### regex_findall

| N | python:pure | codon:pure | python:re2 | python:polars |
|---:|---:|---:|---:|---:|
| 100,000 | 26.84 ms (1.0x) | 8.13 ms (3.3x) | 51.93 ms (0.5x) | 3.98 ms (6.7x) |
| 1,000,000 | 271.03 ms (1.0x) | 41.86 ms (6.5x) | 537.26 ms (0.5x) | 37.46 ms (7.2x) |
| 3,000,000 | 813.22 ms (1.0x) | 129.95 ms (6.3x) | 1.64 s (0.5x) | 112.40 ms (7.2x) |

### regex_sub

| N | python:pure | codon:pure | python:re2 | python:polars |
|---:|---:|---:|---:|---:|
| 100,000 | 14.89 ms (1.0x) | 39.95 ms (0.4x) | 155.01 ms (0.1x) | 6.09 ms (2.4x) |
| 1,000,000 | 159.04 ms (1.0x) | 99.18 ms (1.6x) | 1.58 s (0.1x) | 58.37 ms (2.7x) |
| 3,000,000 | 468.20 ms (1.0x) | 322.08 ms (1.5x) | 4.69 s (0.1x) | 175.79 ms (2.7x) |

### str_methods

| N | python:pure | codon:pure |
|---:|---:|---:|
| 100,000 | 3.85 ms (1.0x) | 1.83 ms (2.1x) |
| 1,000,000 | 42.22 ms (1.0x) | 15.59 ms (2.7x) |
| 3,000,000 | 128.16 ms (1.0x) | 39.27 ms (3.3x) |

### levenshtein

| N | python:pure | codon:pure | python:rapidfuzz |
|---:|---:|---:|---:|
| 1,000 | 115.04 ms (1.0x) | 21.66 ms (5.3x) | 41 us (2805.8x) |
| 3,000 | 1.21 s (1.0x) | 179.40 ms (6.7x) | 286 us (4230.0x) |
| 10,000 | 14.28 s (1.0x) | 1.37 s (10.4x) | 2.91 ms (4914.3x) |

## 5. Calling rapidfuzz from Codon in a tight loop: Python interop vs direct C/C++

M string pairs (len 32, ~30% edits), sum of Levenshtein distances. `codon:py_*` use `from python import rapidfuzz` (CPython C-API: str->PyUnicode copies, arg tuple, PyObject_Call, result unboxing). `codon:c_shim*` call rapidfuzz-cpp through a 5-line `extern "C"` shim via `from C import`. `codon:py_noop` calls Python `len()` = pure per-call interop overhead.

### levenshtein_pairs

| N | python:loop | python:cpdist | python:cpdist_mt | codon:py_loop | codon:py_preconv | codon:py_noop | codon:py_cpdist | codon:c_shim | codon:c_shim_par | codon:native_dp |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 25.72 ms (1.0x) | 94.20 ms (0.3x) | 82.45 ms (0.3x) | 83.71 ms (0.3x) | 49.75 ms (0.5x) | 51.92 ms (0.5x) | 70.69 ms (0.4x) | 13.91 ms (1.8x) | 5.97 ms (4.3x) | 176.84 ms (0.1x) |
| 1,000,000 | 255.94 ms (1.0x) | 368.23 ms (0.7x) | 212.99 ms (1.2x) | 4.40 s (0.1x) | 680.98 ms (0.4x) | 4.25 s (0.1x) | 309.56 ms (0.8x) | 142.82 ms (1.8x) | 23.91 ms (10.7x) | 1.88 s (0.1x) |
| 3,000,000 | 767.01 ms (1.0x) | 976.34 ms (0.8x) | 494.39 ms (1.6x) | 42.44 s (0.0x) | 1.96 s (0.4x) | 41.40 s (0.0x) | 840.62 ms (0.9x) | 429.93 ms (1.8x) | 71.11 ms (10.8x) | 5.83 s (0.1x) |

## 6. Calling common Python data libraries in a tight loop: CPython vs Codon `from python import`

N = number of calls; inputs cycle over a pool of 1000 prebuilt values; attribute lookups hoisted. `codon:conv` passes native Codon values (converted per call), `codon:pre` passes pre-converted pyobjs, `codon:conv_bigheap` = conv with 3M extra live Codon strings on the GC heap.

### np_sqrt

| N | python:loop | codon:conv | codon:pre | codon:conv_bigheap |
|---:|---:|---:|---:|---:|
| 100,000 | 19.46 ms (1.0x) | 91.09 ms (0.2x) | 92.94 ms (0.2x) | 764.06 ms (0.0x) |
| 1,000,000 | 207.64 ms (1.0x) | 914.83 ms (0.2x) | 920.87 ms (0.2x) | 7.73 s (0.0x) |

### np_dot

| N | python:loop | codon:conv | codon:pre | codon:conv_bigheap |
|---:|---:|---:|---:|---:|
| 100,000 | 62.37 ms (1.0x) | 157.76 ms (0.4x) | 158.49 ms (0.4x) | 825.96 ms (0.1x) |
| 1,000,000 | 632.19 ms (1.0x) | 1.55 s (0.4x) | 1.58 s (0.4x) | 8.43 s (0.1x) |

### pd_timestamp

| N | python:loop | codon:conv | codon:pre | codon:conv_bigheap |
|---:|---:|---:|---:|---:|
| 100,000 | 88.81 ms (1.0x) | 285.53 ms (0.3x) | 286.21 ms (0.3x) | 1.63 s (0.1x) |
| 1,000,000 | 895.84 ms (1.0x) | 2.93 s (0.3x) | 2.89 s (0.3x) | 16.70 s (0.1x) |

### pl_getitem

| N | python:loop | codon:conv | codon:pre | codon:conv_bigheap |
|---:|---:|---:|---:|---:|
| 100,000 | 23.54 ms (1.0x) | 105.89 ms (0.2x) | 105.17 ms (0.2x) | 790.70 ms (0.0x) |
| 1,000,000 | 232.17 ms (1.0x) | 1.05 s (0.2x) | 1.04 s (0.2x) | 7.91 s (0.0x) |

### json_loads

| N | python:loop | codon:conv | codon:pre | codon:conv_bigheap |
|---:|---:|---:|---:|---:|
| 100,000 | 126.34 ms (1.0x) | 351.44 ms (0.4x) | 347.15 ms (0.4x) | 1.71 s (0.1x) |
| 1,000,000 | 1.27 s (1.0x) | 3.48 s (0.4x) | 3.45 s (0.4x) | 17.18 s (0.1x) |

### orjson_loads

| N | python:loop | codon:conv | codon:pre | codon:conv_bigheap |
|---:|---:|---:|---:|---:|
| 100,000 | 38.21 ms (1.0x) | 210.74 ms (0.2x) | 207.85 ms (0.2x) | 1.57 s (0.0x) |
| 1,000,000 | 380.10 ms (1.0x) | 2.09 s (0.2x) | 2.09 s (0.2x) | 15.77 s (0.0x) |

### re_findall

| N | python:loop | codon:conv | codon:pre | codon:conv_bigheap |
|---:|---:|---:|---:|---:|
| 100,000 | 75.09 ms (1.0x) | 164.82 ms (0.5x) | 161.78 ms (0.5x) | 836.19 ms (0.1x) |
| 1,000,000 | 749.64 ms (1.0x) | 1.62 s (0.5x) | 1.61 s (0.5x) | 8.43 s (0.1x) |

## 7. Geo: are 3M points inside any of 200 polygons? shapely (CPython) vs GEOS C API / native Codon

200 star-shaped polygons (16-63 vertices) on a 1000x1000 map; ~21% of points are inside. `python:shapely_*` = shapely 2.2 / GEOS 3.14 from CPython (`loop` = per-point Python loop with bbox prefilter + `contains_xy` on prepared polygons, `union_loop` = one call per point on the prepared union, `vec` = one vectorized call, `strtree` = STRtree query). `python:pure`/`codon:pure` = identical ray-casting source (`geo_bench.py`). `codon:geos*` call libgeos_c directly via `from C import`. Grid = 100x100 cells: per cell 'fully inside a polygon' flag or CSR list of candidate polygons (grid build is inside the timed region). GPU times include host->device copies.

### pip_naive

| N | python:shapely_loop | python:pure | python:shapely_union_loop | python:shapely_vec | python:shapely_strtree | codon:pure | codon:geos | codon:geos_par | codon:pure_par | codon:gpu |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 600.56 ms (1.0x) | 601.16 ms (1.0x) | 289.09 ms (2.1x) | 91.27 ms (6.6x) | 68.51 ms (8.8x) | 41.01 ms (14.6x) | 43.64 ms (13.8x) | 14.61 ms (41.1x) | 9.63 ms (62.4x) | 2.31 ms (259.9x) |
| 1,000,000 | 5.95 s (1.0x) | 5.95 s (1.0x) | 2.45 s (2.4x) | 551.46 ms (10.8x) | 640.45 ms (9.3x) | 347.37 ms (17.1x) | 421.40 ms (14.1x) | 53.75 ms (110.7x) | 50.98 ms (116.8x) | 18.66 ms (318.9x) |
| 3,000,000 | 17.80 s (1.0x) | 17.89 s (1.0x) | 7.31 s (2.4x) | 1.58 s (11.2x) | 1.91 s (9.3x) | 987.81 ms (18.0x) | 1.23 s (14.5x) | 134.47 ms (132.4x) | 141.27 ms (126.0x) | 54.63 ms (325.9x) |

### pip_grid

| N | python:shapely_grid | python:pure_grid | codon:pure_grid | codon:geos_grid | codon:grid_par | codon:grid_gpu |
|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 159.27 ms (1.0x) | 151.00 ms (1.1x) | 21.78 ms (7.3x) | 17.49 ms (9.1x) | 4.67 ms (34.1x) | 2.93 ms (54.4x) |
| 1,000,000 | 1.31 s (1.0x) | 1.21 s (1.1x) | 76.34 ms (17.2x) | 73.89 ms (17.8x) | 11.37 ms (115.4x) | 8.99 ms (145.8x) |
| 3,000,000 | 3.71 s (1.0x) | 3.55 s (1.0x) | 182.86 ms (20.3x) | 204.31 ms (18.2x) | 20.43 ms (181.6x) | 21.41 ms (173.3x) |

## 8. Analysis

### 8.1 Classical algorithms and acceleration (sections 1–2)
- Same source compiled by Codon: 15–35×, from static typing, unboxed ints/floats, and no interpreter dispatch.
- **SIMD / AVX:** no special syntax needed. With flat `Ptr` buffers instead of nested lists, LLVM auto-vectorizes for
  the host CPU (AVX2). `simd.Vec[int,4]` offers explicit vectors. AVX2 vs `--disable-native` (SSE2) on the
  cache-blocked kernel: 3× at N=1600. Vectorizing a memory-bound loop alone gives only 1.4–2.7×.
- **Multithreading (`@par`, OpenMP, no GIL):** N-Queens split by first-two-row placements → 8.7× on 8 cores.
  Naive `@par` Floyd–Warshall is memory-bound (1.8× at N=3200); after 64×64 cache-blocking, `@par` → 23×.
- **GPU (`@gpu.kernel`, CUDA):** 44× vs Codon serial (N=3200). Pitfall: list/array kernel arguments are copied
  host↔device on *every launch*. Allocate once with `gpu.Memory._alloc` and pass a raw `Ptr` instead.
  Needs `--libdevice`.

### 8.2 NumPy / dataframes (section 3)
- Codon ships a native compiled NumPy, but **no native dataframe library** exists for Codon. (Web search was blocked;
  the exaloop GitHub org was checked instead.) Dataframes are used via `from python import polars`
  (+ `import numpy.pybridge` for ndarray transfer).
- Vectorized NumPy has nothing left to compile: Codon ≈ NumPy. Codon pays off when the code is a loop.
- For simple aggregations a native Codon hash group-by beats Polars at 10M rows. Use Polars-via-interop for complex
  relational work (joins, windows, I/O) where the per-call interop overhead is amortized over a large batch.

### 8.3 Text (section 4)
- Codon `str` is Unicode-aware (kind-tagged like CPython). `re` is backed by RE2 (linear time, no backreferences
  or lookarounds).
- Gains are modest because CPython's str/dict/re are already C. `re.findall` 6.5× is the best case. Polars `.str`
  (Rust, multithreaded) is as fast from plain Python. Algorithm choice dominates: rapidfuzz's bit-parallel
  Levenshtein is ~5000× faster than the naive DP in CPython and ~470× faster than the same DP compiled by Codon.

### 8.4 Interop: Codon → Python libraries vs Codon → C/C++ (sections 5–6)
How `from python import X` works: every call goes through the CPython C-API. For each call Codon:
- converts native arguments (e.g. `str` → `PyUnicode_FromKindAndData` copy)
- builds an argument tuple when there is more than one argument, then calls `PyObject_Call`
- checks for an error (`PyErr_Fetch`)
- wraps each returned object or intermediate (`f(x)`, `.value`, `[key]`) in a `pyobj`. A `pyobj` is a
  GC-allocated object with a Boehm finalizer that decrefs it.

So it is the *same C interface CPython uses*, not a faster C/C++ path. Calling C++ directly needs `from C import`
plus an `extern "C"` shim (`interop/rf_shim.cpp`).

**Section 6 answers "is Codon slower when calling Python packages?" — yes.** Per call (N=1M, small heap):

| case | CPython | Codon | extra cost per call in Codon |
|---|---:|---:|---:|
| `np.sqrt(float)` | 0.21 µs | 0.91 µs | +0.7 µs (4.4×) |
| `np.dot(a8, b8)` | 0.63 µs | 1.55 µs | +0.9 µs (2.5×) |
| `pd.Timestamp(str).value` | 0.90 µs | 2.93 µs | +2.0 µs (3.3×) |
| `pl.Series[idx]` | 0.23 µs | 1.05 µs | +0.8 µs (4.5×) |
| `json.loads(s)["a"]` | 1.27 µs | 3.48 µs | +2.2 µs (2.7×) |
| `orjson.loads(s)["a"]` | 0.38 µs | 2.09 µs | +1.7 µs (5.5×) |
| `re.compile(..).findall(s)` | 0.75 µs | 1.62 µs | +0.9 µs (2.2×) |

Findings:
1. **Argument conversion is not the bottleneck.** Pre-converting args to `pyobj` (`codon:pre`) is the same speed
   as converting per call (`codon:conv`).
2. **The cost is Codon's GC bookkeeping for Python objects.** Every result or intermediate becomes a finalizable
   `pyobj`. Boehm GC ran ~244 full collections per 1M calls (measured with `GC_get_gc_no`), and that count does
   not shrink as the heap grows. Each collection marks:
   - the whole live Codon heap;
   - very likely also the data segments of every loaded extension library. This is inferred, not traced:
     importing pandas or polars added 140–180 ns to every call while the GC count stayed the same.

   Measured with Python `len()` in a loop (1M calls):

   | live Codon heap | cost per call |
   |---:|---:|
   | 0 MB | 214 ns |
   | 14 MB | 580 ns |
   | 46 MB | 1.36 µs |
   | 138 MB | 3.8 µs |

3. **Superlinear blow-up:** if the live heap grows with N (e.g. the input strings), total time is O(N × heap) ≈ O(N²).
   This explains section 5, where `codon:py_loop` took 4.4 s at 1M pairs and 42 s at 3M pairs. With 3M extra live
   strings (`codon:conv_bigheap`), every library call costs 7.7–17 µs, i.e. 10–40× slower than CPython.
   CPython uses reference counting, so it has no such cost.
4. **Direct C/C++ is the fast path:** `from C import` costs a plain function call. rapidfuzz-cpp through a 5-line shim:
   - 1.8× faster than CPython's own loop over rapidfuzz (which is C++ inside, but pays the interpreter
     and C-API cost on each call);
   - 10.8× faster with `@par`, which is safe because no GIL is involved.
5. **Batch APIs reduce the gap:** `rapidfuzz.process.cpdist` called once from Codon (0.84 s @3M) is close to CPython
   (0.98 s). Interop overhead is per *call*, not per element.

### 8.5 Codon vs Numba (section 2b) — is Numba strictly better? No.
- **Identical source** (`@njit` applied to the unchanged `algos.py` functions with lists and recursion): parity at
  large N. FW 1600: 4.56 s vs 4.09 s; N-Queens 15: 1.18 s vs 1.24 s; LCS 20k: 1.26 s vs 1.26 s. Both are LLVM backends.
- **Optimized Floyd–Warshall (CPU): Numba wins every variant.**

  | variant (N=3200) | Codon | Numba | Numba faster by |
  |---|---:|---:|---:|
  | flat | 23.7 s | 13.0 s | 1.8× |
  | `@par` / `prange` | 18.6 s | 7.3 s | 2.6× |
  | blocked | 7.18 s | 5.84 s | 1.2× |
  | blocked + par | 1.36 s | 1.16 s | 1.17× |

  For the plain parallel loop, using a conditional store instead of a select brings Codon from 1.33 s to 0.73 s at
  N=1600, still behind Numba's 0.49 s. The flat serial gap stayed unexplained (likely aliasing or vectorization
  codegen for `Ptr` row pointers).
- **N-Queens (recursive integer bit tricks): Codon wins.** Serial 7.48 s vs 8.25 s; parallel 0.85 s vs 0.97 s at N=16.
- **GPU: identical.** Floyd–Warshall 750 ms vs 758 ms; N-Queens 686 ms vs 689 ms. Both emit PTX via NVVM, and the
  kernels are the same.
- **Numba costs and gotchas:**
  - JIT compile is 0.3–1.1 s per function per process, unless `cache=True`. Codon instead takes 5–12 s ahead of time
    per binary.
  - The default `omp` threading layer (libgomp) was **2–5× slower** for many short `prange` regions. Use `workqueue`.
  - With 2-D indexing over `range(jb, j1)`, the loop kept negative-index checks and did not vectorize (3× slower).
    Zero-based row slices fixed it.
  - Numba CUDA needs the separate `numba-cuda` package. Installing it downgraded NumPy from 2.5.4 to 2.4.6.
- **Verdict:** for NumPy-array numeric kernels inside a Python program, Numba is equal or better and easier to
  adopt. Codon wins for general Python code (strings, dicts, classes, recursion, generic containers), for standalone
  binaries, and for calling C libraries with zero overhead. On GPU they are the same.

### 8.6 Geo point-in-polygon (section 7)
- **CPython + shapely in a per-point loop: 17.8 s (5.9 µs per point) at 3M points.** That is the same time as pure-Python
  ray casting (17.9 s): GEOS's own work is negligible, and the interpreter loop over 200 bounding boxes dominates.
  Calling a prepared union once per point gets 7.3 s. The best idiomatic Python is **vectorized** `contains_xy` on
  the prepared union: **1.58 s**. STRtree takes 1.91 s.
- **Codon → GEOS C API (`from C import`, no Python):**
  - 1.23 s serial (14.5× vs the shapely loop).
  - **134 ms with `@par`.** This requires a GEOS context *and its own prepared geometries per thread*; sharing
    prepared polygons across threads crashed (double free).
  - So for C libraries, Codon is the "fast loop host" that CPython cannot be.
- **Pure Codon ray casting (identical source to the Python version):** 0.99 s, 18× vs CPython running the same source.
  That beats GEOS here: no call overhead, and the polygons are simple. With `@par` 141 ms; GPU **55 ms** (326×).
- **Grid index (100×100 cells; each cell is either "fully inside a polygon" or lists its candidate polygons; build
  time included):**

  | implementation | 3M points |
  |---|---:|
  | Python, pure | 3.55 s (5× faster than naive Python, still slower than shapely vectorized) |
  | Python, shapely exact test | 3.71 s |
  | Codon | 183 ms |
  | Codon → GEOS | 204 ms |
  | **Codon + `@par`** | **20 ms** |
  | Codon on GPU | 21 ms (bound by copying 48 MB to the device) |

- **Takeaway:** a better algorithm (grid), compiled code and threads multiply: **17.8 s → 20 ms** (870×), and 77×
  faster than the best vectorized shapely.

### 8.7 Recommendations
- Hot loops: write them in Codon (native types). Add `@par` and cache-blocking for big wins; use GPU for regular
  data-parallel kernels.
- Need a C/C++ library in a hot loop: `from C import` (with an `extern "C"` shim for C++). Don't use the Python binding.
- Need a Python library: call it **coarsely** (whole arrays or frames per call). Never call it per element in a tight loop,
  especially while holding large Codon data structures.
- Pure vectorized NumPy/Polars code gains nothing from Codon.
- Numeric NumPy kernels in an existing Python app: try Numba first (no build step, equal or faster). Choose Codon for
  non-array Python code, recursion-heavy integer code, standalone binaries, or direct C-library loops.
- Spatial workloads: add a spatial index (grid) first, then compile with Codon and add `@par`. Use the GEOS C API
  directly for exact predicates on complex geometries, with per-thread contexts and per-thread prepared geometries.

## 9. Gotchas encountered
- **Codon `-release` GC-safety bug (found here):** a pool list held only in a local and indexed as `xs[i % P]` inside
  a loop calling Python was freed by the GC mid-loop. Symptoms were wrong results and segfaults after the first
  collection. The debug build was correct, and adding a `print` hid the bug. Workaround: keep such data
  in module-level globals (always GC roots). See `interop/libs_codon.codon`.
- Codon compiled a function that reused one local name for lists of different types in separate `if` branches, but
  the program read garbage. Use distinct names.
- LLVM removes repeated identical pure calls (common-subexpression elimination), which gave 0 s timings. Time one
  call per process.
- Codon dicts are homogeneously typed. Build Polars frames from a list of `pl.Series`.
- Codon's numpy bridge imports matplotlib if present; conda's matplotlib needed a newer libstdc++ → removed it.
- Linker needs `-lz` → zlib in env plus `LIBRARY_PATH`. `from python import` needs `CODON_PYTHON` → libpython.
- GPU: pass raw device `Ptr`s for iterative kernels. Build with `--libdevice=$CODON_LIBDEVICE`.
- `GC_PRINT_STATS=1` works only with the env activated (libpython lookup).
- GEOS from Codon `@par`: prepared geometries shared between threads crashed (double free). Use a per-thread
  `GEOS_init_r` context and per-thread `GEOSPrepare_r` copies.
- Numba: use the `workqueue` threading layer, zero-based slices for vectorization, and `numba-cuda` for CUDA.
  Installing it downgraded NumPy from 2.5.4 to 2.4.6; sections 1–6 ran on 2.5.4.

## Files
`algo/` (classical + accel + `algos_numba.py`), `geo/` (`geo_bench.py` identical source, `geo_shapely.py`, `geo_codon.codon`), `dataframe/`, `text/`, `interop/` (`interop_bench.codon`, `rf_shim.cpp`, `libs_py.py`,
`libs_codon.codon`), `run_benchmarks.py`, `results/*.csv`, `results/RESULTS.md`.
