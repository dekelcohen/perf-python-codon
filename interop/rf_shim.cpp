// Thin C ABI over the header-only rapidfuzz-cpp library so Codon can call it via `from C import`.
#include <cstdint>
#include <string_view>
#include <rapidfuzz/distance/Levenshtein.hpp>

extern "C" int64_t rf_lev(const char* a, int64_t na, const char* b, int64_t nb) {
    return rapidfuzz::levenshtein_distance(std::string_view(a, na), std::string_view(b, nb));
}
