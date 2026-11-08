# Codon vs Python — runtime research
* TODO: Measure C++/C
* TODO: Measure Memory + operate in Batches
* TODO: 
Compares [Codon](https://github.com/exaloop/codon) (AOT Python-subset compiler, LLVM) with CPython on
(1) classical CPU-heavy algorithms, (2) accelerating them in Codon with SIMD / multithreading / GPU,
(3) NumPy / dataframe-style workloads, (4) text / strings / regex and (5–6) the cost of calling Python
libraries (and C/C++) from Codon in tight loops.

**Full write-up (starts with a when-to-use-Codon table): [`benchmark_report.md`](benchmark_report.md).**

Results: [`results/RESULTS.md`](results/RESULTS.md) (raw: `results/<suite>.csv`).

## Key findings (i7-10700 8C/16T + RTX 3070, see RESULTS.md)

- **Pure-Python algorithms**: same source, Codon is **~15–35× faster** (Floyd–Warshall N=800: 15.3 s → 0.48 s;
  N-Queens 14: 7.4 s → 0.21 s; LCS 10k: 7.5 s → 0.35 s).
- **Further acceleration inside Codon** (vs the Codon baseline above, so multiply by ~20–30× for vs CPython):
  - **SIMD/AVX**: Codon targets the host CPU by default, so LLVM auto-vectorizes with AVX2 once loops use flat
    `Ptr` buffers (no list bounds checks). Explicit `simd.Vec[int,4]` adds a little more. AVX2 vs generic SSE2
    on the cache-blocked kernel: **3.0×** (0.88 s vs 2.73 s @N=1600).
  - **Multithreading (`@par`, OpenMP)**: N-Queens split over first-two-row placements → **8.7×** on 8 cores.
    Floyd–Warshall: naive `@par` only 1.8× at N=3200 (memory bound); **cache-blocked + `@par` → 23×**
    (33.3 s → 1.42 s).
  - **GPU (`@gpu.kernel`, CUDA)**: Floyd–Warshall **44×** (33.3 s → 0.75 s @N=3200), i.e. ~**1000×+ vs CPython**.
  - LCS DP has a serial dependency per row; it would need an anti-diagonal wavefront or a bit-parallel
    algorithm, not tested.
- **Vectorized NumPy**: Codon-NumPy ≈ CPython NumPy (both call compiled kernels). A Codon `@par` loop gives
  ~5.4× at 10M elements by using all cores.
- **Python loops over arrays**: Codon ~52× faster than CPython; numba is still ~1.8× faster than Codon here.
- **Group-by**: in Codon, a native compiled hash group-by beats Polars-called-from-Codon at large N
  (interop + array conversion overhead), and is on par with or faster than Polars called from Python.
  So for Codon: **use native code for simple aggregations; call Polars via interop for complex queries**
  (joins, window functions, I/O) where rewriting isn't worth it.
- **Text / regex**: gains are much smaller than for numeric code — Codon **1.5–3×** on dict word counting,
  `re.sub` and str methods (slower than CPython at 100k words), **~6.5×** on `re.findall` (RE2) and
  **5–10×** on char-level Levenshtein DP. Polars `.str` regex (Rust, multithreaded) matches or beats Codon from
  plain Python; `google-re2`'s Python binding is *slower* than stdlib `re` here. Specialized C++ libraries
  with better algorithms (`rapidfuzz`, bit-parallel Levenshtein) beat any compiled naive DP by 1000×+.
- **Calling Python libraries from Codon** (`from python import`) goes through the CPython C-API. Each call is
  **2–5.5× slower than the same loop in CPython** (numpy, pandas, polars, json, orjson, re, rapidfuzz). Every
  returned object is wrapped in a finalizable `pyobj`, which triggers frequent Boehm GC collections that scan
  the whole Codon heap. With a large live heap this becomes 10–40× slower and **superlinear in N**. Calling
  C/C++ directly (`from C import` + an `extern "C"` shim) is 1.8× faster than CPython, or 10.8× with `@par`.
  Rule: call Python libraries coarsely (whole arrays per call), never per element.

## Environment

Conda env `codon_eval` (Python 3.12, numpy, numba, pandas, polars, pyarrow, zlib, google-re2, rapidfuzz,
orjson, rapidfuzz-cpp, shapely/GEOS, numba-cuda, tbb, cuda-nvvm-tools 12.2) with Codon 0.20.6 installed inside the env at
`$CONDA_PREFIX/opt/codon-deploy-linux-x86_64`. Activation hooks
(`$CONDA_PREFIX/etc/conda/activate.d/codon.sh`) set:

- `PATH` -> codon binary
- `CODON_PYTHON` -> env's `libpython3.12.so` (needed for `from python import ...`)
- `LIBRARY_PATH` -> env `lib/` (codon's linker needs `-lz`)
- `CODON_LIBDEVICE` -> env's CUDA `libdevice.10.bc` (needed to build `@gpu.kernel` code:
  `codon build -release --libdevice=$CODON_LIBDEVICE ...`)

`codon-jit` (pip) is also installed for `@codon.jit` use from Python.

```bash
conda activate codon_eval
cd research
python run_benchmarks.py                     # all suites (~30 min)
python run_benchmarks.py --suite text accel  # suites: algo accel numba dataframe text interop interop_libs geo
python run_benchmarks.py --quick             # 2 smallest sizes per bench (sanity check)
```

## Layout

| path | what |
|---|---|
| `algo/algos.py` | Floyd–Warshall O(N³), N-Queens bitmask backtracking, LCS O(N²) DP. **Identical source** run by `python` and compiled by `codon build -release`. |
| `algo/algos_accel.codon` | Codon-only accelerated variants: flat buffers (auto-vectorized AVX2), explicit `simd.Vec`, `@par` (OpenMP), cache-blocked tiles, CUDA `@gpu.kernel`; N-Queens split across threads. Also built with `--disable-native` to isolate AVX2. |
| `algo/algos_numba.py` | Numba counterparts: `@njit` on the unchanged `algos.py` functions, plus prange / blocked / `numba.cuda` variants (incl. N-Queens GPU, also in `algos_accel.codon`). |
| `geo/geo_bench.py` | Point-in-polygon (3M points × 200 polygons), naive and 100×100 grid index. **Identical source** for CPython / Codon. |
| `geo/geo_shapely.py`, `geo/geo_codon.codon` | shapely (loop / union / vectorized / STRtree / grid) vs Codon → GEOS C API (`-l geos_c`), `@par`, GPU. |
| `dataframe/bench_py.py` | Python: NumPy, numba, pandas, Polars. |
| `dataframe/bench_codon.codon` | Codon: native Codon-NumPy, `@par` multithreaded, and Polars via `from python import polars`. |
| `text/text_bench.py` | Word frequency, regex findall / sub, str methods, Levenshtein. **Identical source** for CPython and Codon. |
| `text/text_py_fast.py` | Optimized Python baselines: `Counter`, `google-re2`, Polars `.str` ops, `rapidfuzz`. |
| `interop/interop_bench.codon`, `interop_py.py`, `rf_shim.cpp` | rapidfuzz from Codon: Python bridge vs C++ via `from C import` shim (`librf_shim.so`, built by the runner with g++). |
| `interop/libs_codon.codon`, `libs_py.py` | Tight loops calling numpy / pandas / polars / json / orjson / re: CPython vs Codon `from python import`. |
| `run_benchmarks.py` | Per-suite config; builds, runs every (bench, impl, N) as best-of-3 processes, checks all checksums agree, writes the report. |

NumPy / dataframe workloads (data is deterministic and identical on both sides):

- `numpy_vec` – `sum(sin(x)*exp(-x)+sqrt(x))`: vectorized NumPy (the best case for plain Python).
- `numpy_loop` – sequential EMA with a data-dependent reset: can't be vectorized; the typical "Python loop over an array" hot spot.
- `groupby` – sum / mean / count / max over 1000 integer keys (dataframe group-by).

## Dataframe libraries for Codon — findings

- Codon ships a **native, compiled NumPy** (`import numpy as np` in Codon), but there is **no native Codon
  dataframe library** (no Codon pandas/Polars port in exaloop's repos/docs as of Codon 0.20.6).
- Therefore Codon uses dataframes via Python interop: `from python import polars as pl`. Arrays cross the
  boundary with `import numpy.pybridge` (`pl.Series("v", codon_ndarray)`; back via
  `np.ndarray[float,1].__from_py__(pyobj.p)`).
- Polars was chosen over pandas as the interop target since it's Rust/multithreaded and far faster than pandas.
  The benchmark also includes a hand-written native Codon hash group-by, so you can see whether
  "Polars from Codon" or "plain compiled loops" wins at each N (see `groupby` table).

## Gotchas found while building this

- Codon dicts are homogeneously typed: `pl.DataFrame({"k": int_arr, "v": float_arr})` doesn't compile;
  build from a list of `pl.Series` instead.
- Repeating a pure native call in-process for timing lets LLVM CSE the repeats (0 s timings). Each native
  measurement is therefore a single call per process; the runner repeats processes.
- Codon's numpy bridge imports matplotlib if present; conda's matplotlib needs a newer `libstdc++` than the
  system one Codon executables load, so matplotlib was removed from the env.
- Codon `str` is Unicode-aware (kind-tagged like CPython) and `re` is RE2 (linear time, no backreferences /
  lookarounds). Text benchmarks use ASCII-only data.
- GPU kernels: passing lists/arrays to a `@gpu.kernel` copies host↔device on **every launch**. For iterative
  kernels (Floyd–Warshall launches N kernels) allocate once with `gpu.Memory._alloc` and pass a raw `Ptr`.
- CPU `@par` on Floyd–Warshall barely helps until the algorithm is cache-blocked: the plain version is
  memory-bandwidth bound once the N×N int64 matrix exceeds L3.
- Codon `-release` GC-safety bug: a list held only in a local, indexed `xs[i % P]` in a loop that calls Python,
  was freed mid-loop (wrong results / segfaults; debug build fine). Workaround: keep it in a module global.
- Reusing one local name for different list types in separate `if` branches compiled but produced garbage.
