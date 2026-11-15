#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>

namespace {

using i128 = __int128_t;
using u128 = __uint128_t;

struct U256 {
  std::array<uint64_t, 4> limb{};
};

U256 multiply_u64(U256 value, uint64_t factor) {
  u128 carry = 0;
  for (size_t index = 0; index < value.limb.size(); ++index) {
    const u128 product = static_cast<u128>(value.limb[index]) * factor + carry;
    value.limb[index] = static_cast<uint64_t>(product);
    carry = product >> 64;
  }
  return value;
}

U256 exact_product(std::initializer_list<uint64_t> factors) {
  U256 value;
  value.limb[0] = 1;
  for (const uint64_t factor : factors) {
    value = multiply_u64(value, factor);
  }
  return value;
}

int compare_u256(const U256& left, const U256& right) {
  for (int index = 3; index >= 0; --index) {
    if (left.limb[index] != right.limb[index]) {
      return left.limb[index] < right.limb[index] ? -1 : 1;
    }
  }
  return 0;
}

constexpr int64_t kScale = int64_t{1} << 18;
bool reciprocal_precision(int64_t x, int64_t y, int bits, bool inclusive) {
  if (x <= 0 || y <= 0 || (bits != 12 && bits != 13)) {
    return false;
  }
  i128 difference = static_cast<i128>(x) * static_cast<i128>(y) -
                    static_cast<i128>(kScale) * kScale;
  if (difference < 0) {
    difference = -difference;
  }
  const i128 threshold = i128{1} << (36 - bits);
  return inclusive ? difference <= threshold : difference < threshold;
}

bool rsqrt_precision(int64_t x, int64_t y, int bits, bool inclusive) {
  if (x <= 0 || y <= 0) {
    return false;
  }
  const uint64_t m = uint64_t{1} << bits;
  const U256 middle = exact_product(
      {m, m, static_cast<uint64_t>(x), static_cast<uint64_t>(y),
       static_cast<uint64_t>(y)});
  const U256 lower = exact_product(
      {m - 1, m - 1, static_cast<uint64_t>(kScale),
       static_cast<uint64_t>(kScale), static_cast<uint64_t>(kScale)});
  const U256 upper = exact_product(
      {m + 1, m + 1, static_cast<uint64_t>(kScale),
       static_cast<uint64_t>(kScale), static_cast<uint64_t>(kScale)});
  const int lower_order = compare_u256(lower, middle);
  const int upper_order = compare_u256(middle, upper);
  return inclusive ? lower_order <= 0 && upper_order <= 0
                   : lower_order < 0 && upper_order < 0;
}

}

extern "C" int cdi_site_f18_precision_batch(
    const int64_t* operands, const int64_t* outputs, int64_t rows,
    int operator_code, int bits, int inclusive, uint8_t* passed) {
  if (operands == nullptr || outputs == nullptr || passed == nullptr ||
      rows < 0 || (operator_code != 0 && operator_code != 1) ||
      (bits != 12 && bits != 13) || (inclusive != 0 && inclusive != 1)) {
    return -1;
  }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t row = 0; row < rows; ++row) {
    const bool ok = operator_code == 0
                        ? reciprocal_precision(
                              operands[row], outputs[row], bits, inclusive != 0)
                        : rsqrt_precision(
                              operands[row], outputs[row], bits, inclusive != 0);
    passed[row] = static_cast<uint8_t>(ok);
  }
  return 0;
}
