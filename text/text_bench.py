# Text / string / regex benchmarks. The SAME source runs under CPython and Codon.
#   python text_bench.py <bench> <N>
#   codon build -release -o text_codon text_bench.py && ./text_codon <bench> <N>
# bench: word_freq | regex_findall | regex_sub | str_methods  (N = number of words)
#        levenshtein                                          (N = string length)
# Output: RESULT,<bench>,<N>,<seconds>,<checksum>
# Note: Codon str is a byte string (ASCII); the generated text is ASCII-only.
import re
import sys
import time

SYLLABLES = ["ka", "to", "re", "ing", "mi", "an", "the", "lo", "su", "pe", "ra", "ion", "ed", "ab", "qu", "x"]
HOSTS = ["mail", "corp", "uni", "lab"]
TLDS = ["com", "org", "net"]


def lcg_next(s: int) -> int:
    return (s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFF


def gen_text(n_words: int) -> str:
    s = 2024
    words = []
    for w in range(n_words):
        s = lcg_next(s)
        r = (s >> 20) % 100
        if r < 2:
            s = lcg_next(s)
            words.append(SYLLABLES[(s >> 20) % 16] + SYLLABLES[(s >> 28) % 16] + "." + str((s >> 36) % 100)
                         + "@" + HOSTS[(s >> 24) % 4] + "." + TLDS[(s >> 30) % 3])
        elif r < 7:
            s = lcg_next(s)
            words.append(str((s >> 20) % 1000000))
        else:
            s = lcg_next(s)
            k = 1 + (s >> 20) % 4
            parts = []
            for i in range(k):
                parts.append(SYLLABLES[(s >> (24 + 4 * i)) % 16])
            words.append("".join(parts))
        if w % 12 == 11:
            words.append("\n")
    return " ".join(words)


def gen_pair(n: int):
    s = 99
    a = []
    b = []
    for i in range(n):
        s = lcg_next(s)
        c = "acgt"[(s >> 20) % 4]
        a.append(c)
        # b = a with ~10% mutations
        if (s >> 30) % 10 == 0:
            b.append("acgt"[(s >> 34) % 4])
        else:
            b.append(c)
    return "".join(a), "".join(b)


# ---------------- benchmarks ----------------
def word_freq(text: str) -> int:
    counts = {}
    for w in text.split():
        counts[w] = counts.get(w, 0) + 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    chk = len(counts)
    for kv in top:
        chk += kv[1]
    return chk


EMAIL = re.compile(r"[a-z0-9.]+@[a-z]+\.(?:com|org|net)")
ING = re.compile(r"\b[a-z]*ing\b")
NUM = re.compile(r"\b\d{4,}\b")


def regex_findall(text: str) -> int:
    return len(EMAIL.findall(text)) * 1000000 + len(ING.findall(text)) + len(NUM.findall(text))


def regex_sub(text: str) -> int:
    out = re.sub(r"\d+", "#", text)
    out = re.sub(r"(the|ion)", "_", out)
    return len(out)


def str_methods(text: str) -> int:
    up = text.upper()
    rep = text.replace("ing", "ING")
    chk = up.count("THE") + rep.count("ING") + text.count("ka")
    # manual substring scan with find()
    pos = text.find("quing")
    while pos >= 0:
        chk += 1
        pos = text.find("quing", pos + 1)
    for line in text.split("\n"):
        if line.strip().startswith("ka"):
            chk += 1
    return chk


def levenshtein(a: str, b: str) -> int:
    n = len(a)
    m = len(b)
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            x = prev[j] + 1
            y = cur[j - 1] + 1
            z = prev[j - 1] + cost
            if y < x:
                x = y
            if z < x:
                x = z
            cur[j] = x
        prev, cur = cur, prev
    return prev[m]


def main():
    bench = sys.argv[1]
    n = int(sys.argv[2])
    chk = 0
    if bench == "levenshtein":
        a, b = gen_pair(n)
        t0 = time.perf_counter()
        chk = levenshtein(a, b)
        dt = time.perf_counter() - t0
    else:
        text = gen_text(n)
        t0 = time.perf_counter()
        if bench == "word_freq":
            chk = word_freq(text)
        elif bench == "regex_findall":
            chk = regex_findall(text)
        elif bench == "regex_sub":
            chk = regex_sub(text)
        elif bench == "str_methods":
            chk = str_methods(text)
        else:
            raise ValueError("unknown bench " + bench)
        dt = time.perf_counter() - t0
    print(f"RESULT,{bench},{n},{dt:.6f},{chk}")
if __name__ == "__main__":
    main()
