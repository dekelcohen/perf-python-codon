# CPython + shapely (GEOS) variants of the point-in-polygon benchmark (data/grid from geo_bench.py).
#   python geo_shapely.py <variant> <N>      Output: RESULT,<variant>,<N>,<seconds>,<checksum>
# naive (no spatial index):
#   loop        - Python loop per point; bbox prefilter in Python; shapely.contains_xy(prepared poly, x, y)
#   union_loop  - Python loop per point; ONE contains_xy call per point on the prepared union of all polygons
#   vec         - vectorized: contains_xy(prepared union, xs_array, ys_array) - one call total
#   strtree     - vectorized: STRtree(polys).query(points, predicate="within")
# grid (same grid index as geo_bench.py, built in Python):
#   grid_loop   - Python loop per point; grid lookup; contains_xy per candidate polygon
# Timed region = everything after raw coordinate generation (geometry construction, union, prepare, grid build).
import sys
import time

import numpy as np
import shapely

from geo_bench import build_grid, cell_of, checksum, gen_points, gen_polygons


def make_polys(px, py, off):
    polys = [shapely.Polygon(list(zip(px[off[p]:off[p + 1]], py[off[p]:off[p + 1]]))) for p in range(len(off) - 1)]
    shapely.prepare(polys)
    return polys


def main():
    v, n = sys.argv[1], int(sys.argv[2])
    px, py, off, bx0, by0, bx1, by1 = gen_polygons()
    xs, ys = gen_points(n)
    flags = [0] * n
    t0 = time.perf_counter()
    polys = make_polys(px, py, off)
    contains_xy = shapely.contains_xy
    P = len(polys)
    if v == "loop":
        for i in range(n):
            x = xs[i]
            y = ys[i]
            for p in range(P):
                if bx0[p] <= x <= bx1[p] and by0[p] <= y <= by1[p] and contains_xy(polys[p], x, y):
                    flags[i] = 1
                    break
    elif v == "union_loop":
        u = shapely.union_all(polys)
        shapely.prepare(u)
        for i in range(n):
            if contains_xy(u, xs[i], ys[i]):
                flags[i] = 1
    elif v == "vec":
        u = shapely.union_all(polys)
        shapely.prepare(u)
        flags = contains_xy(u, np.asarray(xs), np.asarray(ys))
    elif v == "strtree":
        tree = shapely.STRtree(polys)
        pts = shapely.points(np.asarray(xs), np.asarray(ys))
        idx = tree.query(pts, predicate="within")[0]
        flags = np.zeros(n, np.int8)
        flags[idx] = 1
    elif v == "grid_loop":
        full, coff, cand = build_grid(px, py, off, bx0, by0, bx1, by1)
        for i in range(n):
            x = xs[i]
            y = ys[i]
            c = cell_of(x, y)
            if full[c]:
                flags[i] = 1
                continue
            for k in range(coff[c], coff[c + 1]):
                p = cand[k]
                if bx0[p] <= x <= bx1[p] and by0[p] <= y <= by1[p] and contains_xy(polys[p], x, y):
                    flags[i] = 1
                    break
    else:
        sys.exit("unknown variant " + v)
    dt = time.perf_counter() - t0
    if isinstance(flags, np.ndarray):
        idx = np.flatnonzero(flags)
        chk = len(idx) * 1000000000 + int((idx % 997).sum())
    else:
        chk = checksum(flags)
    print(f"RESULT,{v},{n},{dt:.6f},{chk}")


if __name__ == "__main__":
    main()
