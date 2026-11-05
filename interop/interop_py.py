# CPython reference for interop_bench.codon (same pairs, same checksum).
#   python interop_py.py <impl> <M>
# impl: loop (Levenshtein.distance per pair) | cpdist (batched, 1 thread) | cpdist_mt (batched, all cores)
import sys
import time

from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cpdist

L = 32


def lcg_next(s):
    return (s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFF


def gen_pairs(m):
    A, B = [], []
    s = 4242
    for _ in range(m):
        a, b = [], []
        for i in range(L):
            s = lcg_next(s)
            c = "acgt"[(s >> 20) % 4]
            a.append(c)
            r = (s >> 30) % 10
            if r == 0:
                b.append("acgt"[(s >> 34) % 4])
            elif r == 1:
                b.append(c + c)
            elif r != 2:
                b.append(c)
        A.append("".join(a))
        B.append("".join(b))
    return A, B


def main():
    impl, m = sys.argv[1], int(sys.argv[2])
    A, B = gen_pairs(m)
    dist = Levenshtein.distance
    t0 = time.perf_counter()
    if impl == "loop":
        total = 0
        for a, b in zip(A, B):
            total += dist(a, b)
    elif impl == "cpdist":
        total = int(cpdist(A, B, scorer=dist, workers=1).sum())
    elif impl == "cpdist_mt":
        total = int(cpdist(A, B, scorer=dist, workers=-1).sum())
    else:
        raise SystemExit("unknown impl " + impl)
    dt = time.perf_counter() - t0
    print(f"RESULT,{impl},{m},{dt:.6f},{total}")


if __name__ == "__main__":
    main()
