# Classical CPU-heavy algorithms. The SAME source runs under CPython and Codon.
#   python algos.py <bench> <N>
#   codon build -release -o algos_codon algos.py && ./algos_codon <bench> <N>
# Output: RESULT,<bench>,<N>,<seconds>,<checksum>
import sys
import time


def lcg_next(s: int) -> int:
    return (s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFF


def floyd_warshall(n: int) -> int:
    """All-pairs shortest paths on a dense random digraph, O(n^3)."""
    INF = 1 << 40
    s = 12345
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        row = dist[i]
        for j in range(n):
            s = lcg_next(s)
            if i == j:
                row[j] = 0
            elif s % 100 < 30:
                row[j] = 1 + (s >> 16) % 1000
    for k in range(n):
        dk = dist[k]
        for i in range(n):
            di = dist[i]
            dik = di[k]
            if dik == INF:
                continue
            for j in range(n):
                nd = dik + dk[j]
                if nd < di[j]:
                    di[j] = nd
    total = 0
    for i in range(n):
        for j in range(n):
            if dist[i][j] < INF:
                total += dist[i][j]
    return total


def _queens(n: int, row: int, cols: int, d1: int, d2: int) -> int:
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


def n_queens(n: int) -> int:
    """Count all N-Queens solutions via bitmask backtracking (exponential)."""
    return _queens(n, 0, 0, 0, 0)


def lcs(n: int) -> int:
    """Longest common subsequence of two random length-n sequences, O(n^2) DP."""
    s = 777
    a = [0] * n
    b = [0] * n
    for i in range(n):
        s = lcg_next(s)
        a[i] = (s >> 20) % 4
        s = lcg_next(s)
        b[i] = (s >> 20) % 4
    prev = [0] * (n + 1)
    cur = [0] * (n + 1)
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = prev[j] if prev[j] > cur[j - 1] else cur[j - 1]
        prev, cur = cur, prev
    return prev[n]


def main():
    bench = sys.argv[1]
    n = int(sys.argv[2])
    t0 = time.time()
    chk = 0
    if bench == "floyd_warshall":
        chk = floyd_warshall(n)
    elif bench == "n_queens":
        chk = n_queens(n)
    elif bench == "lcs":
        chk = lcs(n)
    else:
        raise ValueError("unknown bench " + bench)
    dt = time.time() - t0
    print(f"RESULT,{bench},{n},{dt:.6f},{chk}")


if __name__ == "__main__":
    main()
