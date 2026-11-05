# Point-in-polygon: are N points inside any of 200 polygons? Identical source for CPython and Codon.
#   python geo_bench.py <naive|grid> <N>
#   codon build -release -o geo_same geo_bench.py && ./geo_same <naive|grid> <N>
# Output: RESULT,<bench>,<N>,<seconds>,<checksum>
#   checksum = (#points inside) * 10**9 + sum(i % 997 for inside point i)
# Timed region = everything after raw coordinate generation (grid build included).
import math
import sys
import time

W = 1000.0  # map is [0, W) x [0, W)
NPOLY = 200
G = 100  # grid cells per axis for the optimized version


def lcg_next(s: int) -> int:
    return (s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFF


def gen_polygons():
    """200 simple star-shaped polygons (16..63 vertices, radius 10..40). Flat arrays + offsets + bboxes."""
    s = 4242
    px = [0.0] * 0
    py = [0.0] * 0
    off = [0]
    bx0 = [0.0] * 0
    by0 = [0.0] * 0
    bx1 = [0.0] * 0
    by1 = [0.0] * 0
    for p in range(NPOLY):
        s = lcg_next(s)
        cx = 50.0 + 900.0 * ((s >> 16) / 4294967296.0)
        s = lcg_next(s)
        cy = 50.0 + 900.0 * ((s >> 16) / 4294967296.0)
        s = lcg_next(s)
        r = 10.0 + 30.0 * ((s >> 16) / 4294967296.0)
        s = lcg_next(s)
        nv = 16 + (s >> 16) % 48
        x0, y0, x1, y1 = W, W, 0.0, 0.0
        for k in range(nv):
            s = lcg_next(s)
            ang = 2.0 * math.pi * (k + 0.9 * ((s >> 16) / 4294967296.0)) / nv
            s = lcg_next(s)
            rr = r * (0.5 + 0.5 * ((s >> 16) / 4294967296.0))
            x = cx + rr * math.cos(ang)
            y = cy + rr * math.sin(ang)
            px.append(x)
            py.append(y)
            x0 = min(x0, x)
            y0 = min(y0, y)
            x1 = max(x1, x)
            y1 = max(y1, y)
        off.append(len(px))
        bx0.append(x0)
        by0.append(y0)
        bx1.append(x1)
        by1.append(y1)
    return px, py, off, bx0, by0, bx1, by1


def gen_points(n: int):
    s = 99
    xs = [0.0] * n
    ys = [0.0] * n
    for i in range(n):
        s = lcg_next(s)
        xs[i] = W * ((s >> 16) / 4294967296.0)
        s = lcg_next(s)
        ys[i] = W * ((s >> 16) / 4294967296.0)
    return xs, ys


def pip(px, py, a: int, b: int, x: float, y: float) -> bool:
    """Even-odd ray casting over vertices [a, b)."""
    inside = False
    j = b - 1
    for i in range(a, b):
        yi = py[i]
        yj = py[j]
        if (yi > y) != (yj > y):
            xi = px[i]
            if x < (px[j] - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
        j = i
    return inside


def checksum(flags) -> int:
    cnt = 0
    acc = 0
    for i in range(len(flags)):
        if flags[i]:
            cnt += 1
            acc += i % 997
    return cnt * 1000000000 + acc


def count_naive(px, py, off, bx0, by0, bx1, by1, xs, ys):
    n = len(xs)
    flags = [0] * n
    for i in range(n):
        x = xs[i]
        y = ys[i]
        for p in range(len(off) - 1):
            if bx0[p] <= x <= bx1[p] and by0[p] <= y <= by1[p] and pip(px, py, off[p], off[p + 1], x, y):
                flags[i] = 1
                break
    return flags


# ---------------- grid index ----------------
def _orient(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _seg_cross(ax: float, ay: float, bx: float, by: float, cx: float, cy: float, dx: float, dy: float) -> bool:
    # conservative (touching counts as crossing)
    d1 = _orient(cx, cy, dx, dy, ax, ay)
    d2 = _orient(cx, cy, dx, dy, bx, by)
    d3 = _orient(ax, ay, bx, by, cx, cy)
    d4 = _orient(ax, ay, bx, by, dx, dy)
    return (d1 * d2 <= 0.0) and (d3 * d4 <= 0.0)


def rect_inside(px, py, a: int, b: int, x0: float, y0: float, x1: float, y1: float) -> bool:
    """Rectangle fully inside the simple polygon: 4 corners inside and no polygon edge crosses a rect side."""
    if not (pip(px, py, a, b, x0, y0) and pip(px, py, a, b, x1, y0)
            and pip(px, py, a, b, x0, y1) and pip(px, py, a, b, x1, y1)):
        return False
    j = b - 1
    for i in range(a, b):
        ax = px[j]
        ay = py[j]
        bx = px[i]
        by = py[i]
        j = i
        if max(ax, bx) < x0 or min(ax, bx) > x1 or max(ay, by) < y0 or min(ay, by) > y1:
            continue
        if (_seg_cross(ax, ay, bx, by, x0, y0, x1, y0) or _seg_cross(ax, ay, bx, by, x1, y0, x1, y1)
                or _seg_cross(ax, ay, bx, by, x1, y1, x0, y1) or _seg_cross(ax, ay, bx, by, x0, y1, x0, y0)):
            return False
    return True


def build_grid(px, py, off, bx0, by0, bx1, by1):
    """Per cell: full[c]=1 if some polygon contains the whole cell; else CSR list of candidate polygons
    whose bbox overlaps the cell. Empty cells (most of the map) reject points with one lookup."""
    cs = W / G
    full = [0] * (G * G)
    lists = [[0] * 0 for _ in range(G * G)]
    for p in range(len(off) - 1):
        i0 = int(bx0[p] / cs)
        i1 = min(int(bx1[p] / cs), G - 1)
        j0 = int(by0[p] / cs)
        j1 = min(int(by1[p] / cs), G - 1)
        for cj in range(j0, j1 + 1):
            for ci in range(i0, i1 + 1):
                c = cj * G + ci
                if full[c]:
                    continue
                if rect_inside(px, py, off[p], off[p + 1], ci * cs, cj * cs, (ci + 1) * cs, (cj + 1) * cs):
                    full[c] = 1
                else:
                    lists[c].append(p)
    coff = [0] * (G * G + 1)
    cand = [0] * 0
    for c in range(G * G):
        if not full[c]:
            for p in lists[c]:
                cand.append(p)
        coff[c + 1] = len(cand)
    return full, coff, cand


def cell_of(x: float, y: float) -> int:
    inv = G / W
    return min(int(y * inv), G - 1) * G + min(int(x * inv), G - 1)


def count_grid(px, py, off, bx0, by0, bx1, by1, xs, ys):
    full, coff, cand = build_grid(px, py, off, bx0, by0, bx1, by1)
    n = len(xs)
    flags = [0] * n
    for i in range(n):
        x = xs[i]
        y = ys[i]
        c = cell_of(x, y)
        if full[c]:
            flags[i] = 1
            continue
        for k in range(coff[c], coff[c + 1]):
            p = cand[k]
            if bx0[p] <= x <= bx1[p] and by0[p] <= y <= by1[p] and pip(px, py, off[p], off[p + 1], x, y):
                flags[i] = 1
                break
    return flags


def main():
    bench = sys.argv[1]
    n = int(sys.argv[2])
    px, py, off, bx0, by0, bx1, by1 = gen_polygons()
    xs, ys = gen_points(n)
    t0 = time.perf_counter()
    if bench == "naive":
        flags = count_naive(px, py, off, bx0, by0, bx1, by1, xs, ys)
    elif bench == "grid":
        flags = count_grid(px, py, off, bx0, by0, bx1, by1, xs, ys)
    else:
        print("unknown bench " + bench)
        sys.exit(1)
    dt = time.perf_counter() - t0
    print(f"RESULT,{bench},{n},{dt:.6f},{checksum(flags)}")


if __name__ == "__main__":
    main()
