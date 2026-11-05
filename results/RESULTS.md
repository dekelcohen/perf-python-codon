# Codon vs Python benchmark results

- CPU: Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz (16 threads); GPU: NVIDIA GeForce RTX 3070
- Python 3.12.16, Codon 0.20.6, NumPy 2.4.6, numba 0.68.1, pandas 3.1.1, polars 2.1.0
- Time = best of 3 separate processes; excludes process start, imports and data generation.
- Codon compile time (`codon build -release`): `algos_accel` 11.3s, `algos_accel_generic` 10.1s, `algos_codon` 5.4s, `bench_codon` 13.0s, `geo_same` 5.6s, `geo_codon` 11.7s, `interop_codon` 8.1s, `libs_codon` 8.2s, `text_codon` 10.9s
- Checksums identical across all implementations: **yes**
- Cells: time (speedup vs first column).

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
