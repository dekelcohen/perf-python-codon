# Numba versions of the classical algorithms (counterpart of algos.py / algos_accel.codon).
#   python algos_numba.py <variant> <N>
# Output: RESULT,<variant>,<N>,<seconds>,<checksum>   (JIT compile time excluded: warm-up on a tiny N first;
#         printed separately as "JIT,<variant>,<seconds>" on stderr)
#
# same:<bench>     - @njit applied to the UNCHANGED functions of algos.py (lists, recursion) - fair
#                    "identical source" comparison with CPython / Codon.
# fw_flat          - 2-D int64 ndarray, serial (counterpart of Codon fw_flat)
# fw_par           - prange over rows for each k (Codon fw_par)
# fw_blocked       - 64x64 cache-blocked, serial (Codon fw_blocked)
# fw_blocked_par   - cache-blocked + prange over tiles (Codon fw_blocked_par)
# fw_gpu           - numba.cuda kernel per k, matrix resident on GPU (Codon fw_gpu)
# nq_serial        - recursive bitmask N-Queens (same as algos.py)
# nq_par           - prange over first-two-row placements (Codon nq_par)
# nq_gpu           - one CUDA thread per 4-row prefix, iterative DFS with a per-thread local stack
import functools
import sys
import time

import numpy as np
import numba
from numba import njit, prange

# default 'omp' layer (GNU libgomp) was 2-5x slower here for many short parallel regions (fw_par)
numba.config.THREADING_LAYER = "workqueue"

import algos

INF = 1 << 40
TILE = 64

# ---- identical source: jit the algos.py functions in place (globals resolve to the jitted versions) ----
algos.lcg_next = njit(algos.lcg_next)
algos._queens = njit(algos._queens)
SAME = {
    "floyd_warshall": njit(algos.floyd_warshall),
    "n_queens": njit(algos.n_queens),
    "lcs": njit(algos.lcs),
}
lcg_next = algos.lcg_next


@njit
def make_graph(n):
    d = np.empty((n, n), np.int64)
    s = 12345
    for i in range(n):
        for j in range(n):
            s = lcg_next(s)
            if i == j:
                d[i, j] = 0
            elif s % 100 < 30:
                d[i, j] = 1 + (s >> 16) % 1000
            else:
                d[i, j] = INF
    return d


@njit
def checksum(d):
    t = 0
    for v in d.ravel():
        if v < INF:
            t += v
    return t


@njit
def _relax_row(d, n, i, k):
    dik = d[i, k]
    if dik == INF:
        return
    for j in range(n):
        nd = dik + d[k, j]
        if nd < d[i, j]:
            d[i, j] = nd


@njit
def fw_flat(d):
    n = d.shape[0]
    for k in range(n):
        for i in range(n):
            _relax_row(d, n, i, k)


@njit(parallel=True)
def fw_par(d):
    n = d.shape[0]
    for k in range(n):
        for i in prange(n):
            _relax_row(d, n, i, k)


@njit
def _tile(d, n, ib, jb, kb):
    # zero-based row slices: with range(jb, j1) Numba keeps negative-index wraparound checks
    # and the inner loop does not vectorize (~3x slower)
    i1 = min(ib + TILE, n)
    j1 = min(jb + TILE, n)
    k1 = min(kb + TILE, n)
    m = j1 - jb
    for k in range(kb, k1):
        dk = d[k, jb:j1]
        for i in range(ib, i1):
            dik = d[i, k]
            if dik == INF:
                continue
            di = d[i, jb:j1]
            for j in range(m):
                nd = dik + dk[j]
                di[j] = nd if nd < di[j] else di[j]


@njit
def fw_blocked(d):
    n = d.shape[0]
    nb = (n + TILE - 1) // TILE
    for b in range(nb):
        kb = b * TILE
        _tile(d, n, kb, kb, kb)
        for o in range(nb):
            if o != b:
                _tile(d, n, kb, o * TILE, kb)
                _tile(d, n, o * TILE, kb, kb)
        for bi in range(nb):
            for bj in range(nb):
                if bi != b and bj != b:
                    _tile(d, n, bi * TILE, bj * TILE, kb)


@njit(parallel=True)
def fw_blocked_par(d):
    n = d.shape[0]
    nb = (n + TILE - 1) // TILE
    for b in range(nb):
        kb = b * TILE
        _tile(d, n, kb, kb, kb)
        for t in prange(2 * nb):
            o = t % nb
            if o != b:
                if t < nb:
                    _tile(d, n, kb, o * TILE, kb)
                else:
                    _tile(d, n, o * TILE, kb, kb)
        for t in prange(nb * nb):
            bi = t // nb
            bj = t % nb
            if bi != b and bj != b:
                _tile(d, n, bi * TILE, bj * TILE, kb)


@njit
def _queens(n, row, cols, d1, d2):
    if row == n:
        return 1
    count = 0
    full = (1 << n) - 1
    avail = ~(cols | d1 | d2) & full
    while avail:
        bit = avail & -avail
        avail ^= bit
        count += _queens(n, row + 1, cols | bit, ((d1 | bit) << 1) & full, (d2 | bit) >> 1)
    return count


@njit
def nq_serial(n):
    return _queens(n, 0, 0, 0, 0)


@njit(parallel=True)
def nq_par(n):
    full = (1 << n) - 1
    total = 0
    for c in prange(n * n):
        b0 = 1 << (c // n)
        b1 = 1 << (c % n)
        cols = b0
        d1 = (b0 << 1) & full
        d2 = b0 >> 1
        if b1 & (cols | d1 | d2) == 0:
            total += _queens(n, 2, cols | b1, ((d1 | b1) << 1) & full, (d2 | b1) >> 1)
    return total


# ---------------- GPU ----------------
@functools.lru_cache(None)
def _gpu():
    from numba import cuda, int64

    @cuda.jit
    def fw_kernel(d, n, k):
        j, i = cuda.grid(2)
        if i < n and j < n:
            dik = d[i, k]
            if dik != INF:
                nd = dik + d[k, j]
                if nd < d[i, j]:
                    d[i, j] = nd

    @cuda.jit
    def nq_kernel(n, start, C, D1, D2, out):
        t = cuda.grid(1)
        if t >= C.size:
            return
        full = (1 << n) - 1
        cs = cuda.local.array(32, int64)
        a1 = cuda.local.array(32, int64)
        a2 = cuda.local.array(32, int64)
        av = cuda.local.array(32, int64)
        cs[0] = C[t]
        a1[0] = D1[t]
        a2[0] = D2[t]
        av[0] = ~(C[t] | D1[t] | D2[t]) & full
        sp = 0
        count = 0
        while sp >= 0:
            a = av[sp]
            if a == 0:
                sp -= 1
                continue
            bit = a & -a
            av[sp] = a ^ bit
            if start + sp + 1 == n:
                count += 1
                continue
            c = cs[sp] | bit
            x = ((a1[sp] | bit) << 1) & full
            y = (a2[sp] | bit) >> 1
            sp += 1
            cs[sp] = c
            a1[sp] = x
            a2[sp] = y
            av[sp] = ~(c | x | y) & full
        out[t] = count

    def fw_gpu(d):
        n = d.shape[0]
        B = 16
        G = (n + B - 1) // B
        dd = cuda.to_device(d)
        for k in range(n):
            fw_kernel[(G, G), (B, B)](dd, n, k)
        dd.copy_to_host(d)  # synchronizes

    def nq_gpu(n, depth=4):
        C, D1, D2 = nq_prefixes(n, depth)
        out = cuda.device_array(C.size, np.int64)
        T = 128
        nq_kernel[(C.size + T - 1) // T, T](n, depth, cuda.to_device(C), cuda.to_device(D1), cuda.to_device(D2), out)
        return int(out.copy_to_host().sum())

    return fw_gpu, nq_gpu


@njit
def nq_prefixes(n, depth):
    # all valid placements of the first `depth` rows -> (cols, d1, d2) states
    cap = n ** depth
    C = np.empty(cap, np.int64)
    D1 = np.empty(cap, np.int64)
    D2 = np.empty(cap, np.int64)
    full = (1 << n) - 1
    sc = np.zeros(depth + 1, np.int64)
    s1 = np.zeros(depth + 1, np.int64)
    s2 = np.zeros(depth + 1, np.int64)
    sa = np.zeros(depth + 1, np.int64)
    sa[0] = full
    sp = 0
    m = 0
    while sp >= 0:
        if sp == depth:
            C[m] = sc[sp]
            D1[m] = s1[sp]
            D2[m] = s2[sp]
            m += 1
            sp -= 1
            continue
        a = sa[sp]
        if a == 0:
            sp -= 1
            continue
        bit = a & -a
        sa[sp] = a ^ bit
        c = sc[sp] | bit
        x = ((s1[sp] | bit) << 1) & full
        y = (s2[sp] | bit) >> 1
        sc[sp + 1] = c
        s1[sp + 1] = x
        s2[sp + 1] = y
        sa[sp + 1] = ~(c | x | y) & full
        sp += 1
    return C[:m], D1[:m], D2[:m]


def run(v, n):
    """Return (seconds, checksum) of the timed region."""
    if v.startswith("same:"):
        f = SAME[v[5:]]
        t0 = time.perf_counter()
        c = f(n)
        return time.perf_counter() - t0, c
    if v.startswith("fw_"):
        f = {"fw_flat": fw_flat, "fw_par": fw_par, "fw_blocked": fw_blocked,
             "fw_blocked_par": fw_blocked_par}.get(v) or _gpu()[0]
        d = make_graph(n)
        t0 = time.perf_counter()  # GPU: includes host<->device copies (as Codon fw_gpu)
        f(d)
        dt = time.perf_counter() - t0
        return dt, checksum(d)
    if v.startswith("nq_"):
        f = {"nq_serial": nq_serial, "nq_par": nq_par}.get(v) or _gpu()[1]
        t0 = time.perf_counter()
        c = f(n)
        return time.perf_counter() - t0, c
    sys.exit("unknown variant " + v)


def main():
    v, n = sys.argv[1], int(sys.argv[2])
    warm = {"fw": 70, "nq": 8, "sa": 8}[v[:2]]
    t0 = time.perf_counter()
    run(v, warm)  # JIT compile (+ CUDA context init) on a tiny input
    print(f"JIT,{v},{time.perf_counter() - t0:.3f}", file=sys.stderr)
    dt, chk = run(v, n)
    print(f"RESULT,{v},{n},{dt:.6f},{chk}")


if __name__ == "__main__":
    main()
