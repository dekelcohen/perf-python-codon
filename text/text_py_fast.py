# Optimized-Python baselines for the text benchmarks (same data/checksums as text_bench.py).
#   python text_py_fast.py <bench> <impl> <N>
# word_freq    : counter | polars
# regex_findall: re2 | polars
# regex_sub    : re2 | polars
# levenshtein  : rapidfuzz
# Output: RESULT,<bench>,<impl>,<N>,<seconds>,<checksum>
import sys
import time
from collections import Counter

from text_bench import EMAIL, ING, NUM, gen_pair, gen_text


def word_freq_counter(text):
    counts = Counter(text.split())
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    return len(counts) + sum(c for _, c in top)


def word_freq_polars(text):
    import polars as pl

    words = pl.Series("w", [text]).str.split(" ").explode()
    words = words.filter(words.str.strip_chars() != "").str.strip_chars()
    vc = words.value_counts(name="c").sort(["c", "w"], descending=[True, False])
    return vc.height + int(vc["c"].head(10).sum())


def regex_findall_re2(text):
    import re2

    e, i, n = (re2.compile(p.pattern) for p in (EMAIL, ING, NUM))
    return len(e.findall(text)) * 1000000 + len(i.findall(text)) + len(n.findall(text))


def regex_findall_polars(text):
    import polars as pl

    s = pl.Series("t", text.split("\n"))  # patterns never span a newline
    c = [int(s.str.count_matches(p.pattern).sum()) for p in (EMAIL, ING, NUM)]
    return c[0] * 1000000 + c[1] + c[2]


def regex_sub_re2(text):
    import re2

    out = re2.sub(r"\d+", "#", text)
    out = re2.sub(r"(the|ion)", "_", out)
    return len(out)


def regex_sub_polars(text):
    import polars as pl

    s = pl.Series("t", text.split("\n"))
    s = s.str.replace_all(r"\d+", "#").str.replace_all(r"(the|ion)", "_")
    return int(s.str.len_bytes().sum()) + s.len() - 1  # + joining newlines


def levenshtein_rapidfuzz(ab):
    from rapidfuzz.distance import Levenshtein

    return Levenshtein.distance(ab[0], ab[1])


IMPLS = {
    ("word_freq", "counter"): word_freq_counter,
    ("word_freq", "polars"): word_freq_polars,
    ("regex_findall", "re2"): regex_findall_re2,
    ("regex_findall", "polars"): regex_findall_polars,
    ("regex_sub", "re2"): regex_sub_re2,
    ("regex_sub", "polars"): regex_sub_polars,
    ("levenshtein", "rapidfuzz"): levenshtein_rapidfuzz,
}


def main():
    bench, impl, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    fn = IMPLS[(bench, impl)]
    data = gen_pair(n) if bench == "levenshtein" else gen_text(n)
    fn(data)  # warmup (imports, regex compile caches, thread pools)
    t0 = time.perf_counter()
    chk = fn(data)
    dt = time.perf_counter() - t0
    print(f"RESULT,{bench},{impl},{n},{dt:.6f},{chk}")


if __name__ == "__main__":
    main()
