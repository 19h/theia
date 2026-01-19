#pragma once

// Portable SIMD abstraction layer for psynth
// Supports SSE4.2, AVX2, AVX-512, ARM NEON with automatic fallback to scalar

#include <cstdint>
#include <cmath>
#include <algorithm>

// ============================================================================
// Platform Detection
// ============================================================================

// Detect SIMD capabilities at compile time
#if defined(__AVX512F__) && defined(__AVX512DQ__)
  #define PSYNTH_SIMD_AVX512 1
  #define PSYNTH_SIMD_AVX2 1
  #define PSYNTH_SIMD_SSE42 1
  #define PSYNTH_SIMD_WIDTH 8
  #define PSYNTH_SIMD_NAME "AVX-512"
#elif defined(__AVX2__)
  #define PSYNTH_SIMD_AVX2 1
  #define PSYNTH_SIMD_SSE42 1
  #define PSYNTH_SIMD_WIDTH 4
  #define PSYNTH_SIMD_NAME "AVX2"
#elif defined(__SSE4_2__) || defined(__SSE4_1__) || (defined(_MSC_VER) && defined(__AVX__))
  #define PSYNTH_SIMD_SSE42 1
  #define PSYNTH_SIMD_WIDTH 2
  #define PSYNTH_SIMD_NAME "SSE4.2"
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  #define PSYNTH_SIMD_NEON 1
  #define PSYNTH_SIMD_WIDTH 2
  #define PSYNTH_SIMD_NAME "NEON"
#else
  #define PSYNTH_SIMD_SCALAR 1
  #define PSYNTH_SIMD_WIDTH 1
  #define PSYNTH_SIMD_NAME "Scalar"
#endif

// Include appropriate headers
#if defined(PSYNTH_SIMD_AVX512) || defined(PSYNTH_SIMD_AVX2)
  #include <immintrin.h>
#elif defined(PSYNTH_SIMD_SSE42)
  #include <smmintrin.h>
  #include <emmintrin.h>
#elif defined(PSYNTH_SIMD_NEON)
  #include <arm_neon.h>
#endif

namespace psynth::simd {

// ============================================================================
// Alignment helpers
// ============================================================================

constexpr size_t kCacheLineSize = 64;
constexpr size_t kSimdAlignment = 32;  // AVX alignment

#if defined(_MSC_VER)
  #define PSYNTH_ALIGNED(x) __declspec(align(x))
  #define PSYNTH_ASSUME_ALIGNED(ptr, align) ptr
#else
  #define PSYNTH_ALIGNED(x) __attribute__((aligned(x)))
  #define PSYNTH_ASSUME_ALIGNED(ptr, align) __builtin_assume_aligned(ptr, align)
#endif

#define PSYNTH_CACHE_ALIGNED PSYNTH_ALIGNED(64)
#define PSYNTH_SIMD_ALIGNED PSYNTH_ALIGNED(32)

// Force inline
#if defined(_MSC_VER)
  #define PSYNTH_FORCE_INLINE __forceinline
#else
  #define PSYNTH_FORCE_INLINE inline __attribute__((always_inline))
#endif

// Likely/unlikely branch hints
#if defined(__GNUC__) || defined(__clang__)
  #define PSYNTH_LIKELY(x) __builtin_expect(!!(x), 1)
  #define PSYNTH_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
  #define PSYNTH_LIKELY(x) (x)
  #define PSYNTH_UNLIKELY(x) (x)
#endif

// ============================================================================
// Double4 - 4 packed doubles (AVX or 2x SSE or scalar)
// ============================================================================

#if defined(PSYNTH_SIMD_AVX2) || defined(PSYNTH_SIMD_AVX512)

struct Double4 {
  __m256d v;

  PSYNTH_FORCE_INLINE Double4() : v(_mm256_setzero_pd()) {}
  PSYNTH_FORCE_INLINE Double4(__m256d x) : v(x) {}
  PSYNTH_FORCE_INLINE Double4(double x) : v(_mm256_set1_pd(x)) {}
  PSYNTH_FORCE_INLINE Double4(double a, double b, double c, double d) : v(_mm256_set_pd(d, c, b, a)) {}

  PSYNTH_FORCE_INLINE static Double4 load(const double* ptr) { return _mm256_loadu_pd(ptr); }
  PSYNTH_FORCE_INLINE static Double4 load_aligned(const double* ptr) { return _mm256_load_pd(ptr); }
  PSYNTH_FORCE_INLINE void store(double* ptr) const { _mm256_storeu_pd(ptr, v); }
  PSYNTH_FORCE_INLINE void store_aligned(double* ptr) const { _mm256_store_pd(ptr, v); }

  PSYNTH_FORCE_INLINE Double4 operator+(const Double4& o) const { return _mm256_add_pd(v, o.v); }
  PSYNTH_FORCE_INLINE Double4 operator-(const Double4& o) const { return _mm256_sub_pd(v, o.v); }
  PSYNTH_FORCE_INLINE Double4 operator*(const Double4& o) const { return _mm256_mul_pd(v, o.v); }
  PSYNTH_FORCE_INLINE Double4 operator/(const Double4& o) const { return _mm256_div_pd(v, o.v); }

  PSYNTH_FORCE_INLINE Double4& operator+=(const Double4& o) { v = _mm256_add_pd(v, o.v); return *this; }
  PSYNTH_FORCE_INLINE Double4& operator-=(const Double4& o) { v = _mm256_sub_pd(v, o.v); return *this; }
  PSYNTH_FORCE_INLINE Double4& operator*=(const Double4& o) { v = _mm256_mul_pd(v, o.v); return *this; }

  PSYNTH_FORCE_INLINE Double4 operator<(const Double4& o) const { return _mm256_cmp_pd(v, o.v, _CMP_LT_OQ); }
  PSYNTH_FORCE_INLINE Double4 operator<=(const Double4& o) const { return _mm256_cmp_pd(v, o.v, _CMP_LE_OQ); }
  PSYNTH_FORCE_INLINE Double4 operator>(const Double4& o) const { return _mm256_cmp_pd(v, o.v, _CMP_GT_OQ); }
  PSYNTH_FORCE_INLINE Double4 operator>=(const Double4& o) const { return _mm256_cmp_pd(v, o.v, _CMP_GE_OQ); }

  PSYNTH_FORCE_INLINE Double4 and_not(const Double4& o) const { return _mm256_andnot_pd(o.v, v); }
  PSYNTH_FORCE_INLINE Double4 operator&(const Double4& o) const { return _mm256_and_pd(v, o.v); }
  PSYNTH_FORCE_INLINE Double4 operator|(const Double4& o) const { return _mm256_or_pd(v, o.v); }

  PSYNTH_FORCE_INLINE int movemask() const { return _mm256_movemask_pd(v); }

  PSYNTH_FORCE_INLINE double operator[](int i) const {
    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, v);
    return tmp[i];
  }

  PSYNTH_FORCE_INLINE double horizontal_sum() const {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d sum = _mm_add_pd(lo, hi);
    return _mm_cvtsd_f64(_mm_hadd_pd(sum, sum));
  }
};

PSYNTH_FORCE_INLINE Double4 sqrt(const Double4& x) { return _mm256_sqrt_pd(x.v); }
PSYNTH_FORCE_INLINE Double4 max(const Double4& a, const Double4& b) { return _mm256_max_pd(a.v, b.v); }
PSYNTH_FORCE_INLINE Double4 min(const Double4& a, const Double4& b) { return _mm256_min_pd(a.v, b.v); }
PSYNTH_FORCE_INLINE Double4 abs(const Double4& x) { return _mm256_andnot_pd(_mm256_set1_pd(-0.0), x.v); }

// FMA if available
#if defined(__FMA__)
PSYNTH_FORCE_INLINE Double4 fmadd(const Double4& a, const Double4& b, const Double4& c) {
  return _mm256_fmadd_pd(a.v, b.v, c.v);
}
PSYNTH_FORCE_INLINE Double4 fmsub(const Double4& a, const Double4& b, const Double4& c) {
  return _mm256_fmsub_pd(a.v, b.v, c.v);
}
#else
PSYNTH_FORCE_INLINE Double4 fmadd(const Double4& a, const Double4& b, const Double4& c) {
  return a * b + c;
}
PSYNTH_FORCE_INLINE Double4 fmsub(const Double4& a, const Double4& b, const Double4& c) {
  return a * b - c;
}
#endif

#elif defined(PSYNTH_SIMD_SSE42)

struct Double4 {
  __m128d lo, hi;

  PSYNTH_FORCE_INLINE Double4() : lo(_mm_setzero_pd()), hi(_mm_setzero_pd()) {}
  PSYNTH_FORCE_INLINE Double4(__m128d l, __m128d h) : lo(l), hi(h) {}
  PSYNTH_FORCE_INLINE Double4(double x) : lo(_mm_set1_pd(x)), hi(_mm_set1_pd(x)) {}
  PSYNTH_FORCE_INLINE Double4(double a, double b, double c, double d)
      : lo(_mm_set_pd(b, a)), hi(_mm_set_pd(d, c)) {}

  PSYNTH_FORCE_INLINE static Double4 load(const double* ptr) {
    return Double4(_mm_loadu_pd(ptr), _mm_loadu_pd(ptr + 2));
  }
  PSYNTH_FORCE_INLINE static Double4 load_aligned(const double* ptr) {
    return Double4(_mm_load_pd(ptr), _mm_load_pd(ptr + 2));
  }
  PSYNTH_FORCE_INLINE void store(double* ptr) const {
    _mm_storeu_pd(ptr, lo);
    _mm_storeu_pd(ptr + 2, hi);
  }
  PSYNTH_FORCE_INLINE void store_aligned(double* ptr) const {
    _mm_store_pd(ptr, lo);
    _mm_store_pd(ptr + 2, hi);
  }

  PSYNTH_FORCE_INLINE Double4 operator+(const Double4& o) const {
    return Double4(_mm_add_pd(lo, o.lo), _mm_add_pd(hi, o.hi));
  }
  PSYNTH_FORCE_INLINE Double4 operator-(const Double4& o) const {
    return Double4(_mm_sub_pd(lo, o.lo), _mm_sub_pd(hi, o.hi));
  }
  PSYNTH_FORCE_INLINE Double4 operator*(const Double4& o) const {
    return Double4(_mm_mul_pd(lo, o.lo), _mm_mul_pd(hi, o.hi));
  }
  PSYNTH_FORCE_INLINE Double4 operator/(const Double4& o) const {
    return Double4(_mm_div_pd(lo, o.lo), _mm_div_pd(hi, o.hi));
  }

  PSYNTH_FORCE_INLINE Double4& operator+=(const Double4& o) {
    lo = _mm_add_pd(lo, o.lo);
    hi = _mm_add_pd(hi, o.hi);
    return *this;
  }
  PSYNTH_FORCE_INLINE Double4& operator*=(const Double4& o) {
    lo = _mm_mul_pd(lo, o.lo);
    hi = _mm_mul_pd(hi, o.hi);
    return *this;
  }

  PSYNTH_FORCE_INLINE Double4 operator<(const Double4& o) const {
    return Double4(_mm_cmplt_pd(lo, o.lo), _mm_cmplt_pd(hi, o.hi));
  }
  PSYNTH_FORCE_INLINE Double4 operator>(const Double4& o) const {
    return Double4(_mm_cmpgt_pd(lo, o.lo), _mm_cmpgt_pd(hi, o.hi));
  }

  PSYNTH_FORCE_INLINE Double4 operator&(const Double4& o) const {
    return Double4(_mm_and_pd(lo, o.lo), _mm_and_pd(hi, o.hi));
  }
  PSYNTH_FORCE_INLINE Double4 operator|(const Double4& o) const {
    return Double4(_mm_or_pd(lo, o.lo), _mm_or_pd(hi, o.hi));
  }

  PSYNTH_FORCE_INLINE int movemask() const {
    return _mm_movemask_pd(lo) | (_mm_movemask_pd(hi) << 2);
  }

  PSYNTH_FORCE_INLINE double operator[](int i) const {
    alignas(16) double tmp[4];
    _mm_store_pd(tmp, lo);
    _mm_store_pd(tmp + 2, hi);
    return tmp[i];
  }

  PSYNTH_FORCE_INLINE double horizontal_sum() const {
    __m128d sum = _mm_add_pd(lo, hi);
    return _mm_cvtsd_f64(_mm_hadd_pd(sum, sum));
  }
};

PSYNTH_FORCE_INLINE Double4 sqrt(const Double4& x) {
  return Double4(_mm_sqrt_pd(x.lo), _mm_sqrt_pd(x.hi));
}
PSYNTH_FORCE_INLINE Double4 max(const Double4& a, const Double4& b) {
  return Double4(_mm_max_pd(a.lo, b.lo), _mm_max_pd(a.hi, b.hi));
}
PSYNTH_FORCE_INLINE Double4 min(const Double4& a, const Double4& b) {
  return Double4(_mm_min_pd(a.lo, b.lo), _mm_min_pd(a.hi, b.hi));
}
PSYNTH_FORCE_INLINE Double4 fmadd(const Double4& a, const Double4& b, const Double4& c) {
  return a * b + c;
}
PSYNTH_FORCE_INLINE Double4 fmsub(const Double4& a, const Double4& b, const Double4& c) {
  return a * b - c;
}

#elif defined(PSYNTH_SIMD_NEON)

struct Double4 {
  float64x2_t lo, hi;

  PSYNTH_FORCE_INLINE Double4() : lo(vdupq_n_f64(0)), hi(vdupq_n_f64(0)) {}
  PSYNTH_FORCE_INLINE Double4(float64x2_t l, float64x2_t h) : lo(l), hi(h) {}
  PSYNTH_FORCE_INLINE Double4(double x) : lo(vdupq_n_f64(x)), hi(vdupq_n_f64(x)) {}
  PSYNTH_FORCE_INLINE Double4(double a, double b, double c, double d) {
    double tmp_lo[2] = {a, b};
    double tmp_hi[2] = {c, d};
    lo = vld1q_f64(tmp_lo);
    hi = vld1q_f64(tmp_hi);
  }

  PSYNTH_FORCE_INLINE static Double4 load(const double* ptr) {
    return Double4(vld1q_f64(ptr), vld1q_f64(ptr + 2));
  }
  PSYNTH_FORCE_INLINE void store(double* ptr) const {
    vst1q_f64(ptr, lo);
    vst1q_f64(ptr + 2, hi);
  }

  PSYNTH_FORCE_INLINE Double4 operator+(const Double4& o) const {
    return Double4(vaddq_f64(lo, o.lo), vaddq_f64(hi, o.hi));
  }
  PSYNTH_FORCE_INLINE Double4 operator-(const Double4& o) const {
    return Double4(vsubq_f64(lo, o.lo), vsubq_f64(hi, o.hi));
  }
  PSYNTH_FORCE_INLINE Double4 operator*(const Double4& o) const {
    return Double4(vmulq_f64(lo, o.lo), vmulq_f64(hi, o.hi));
  }
  PSYNTH_FORCE_INLINE Double4 operator/(const Double4& o) const {
    return Double4(vdivq_f64(lo, o.lo), vdivq_f64(hi, o.hi));
  }

  PSYNTH_FORCE_INLINE double operator[](int i) const {
    double tmp[4];
    vst1q_f64(tmp, lo);
    vst1q_f64(tmp + 2, hi);
    return tmp[i];
  }

  PSYNTH_FORCE_INLINE double horizontal_sum() const {
    return vaddvq_f64(lo) + vaddvq_f64(hi);
  }
};

PSYNTH_FORCE_INLINE Double4 sqrt(const Double4& x) {
  return Double4(vsqrtq_f64(x.lo), vsqrtq_f64(x.hi));
}
PSYNTH_FORCE_INLINE Double4 max(const Double4& a, const Double4& b) {
  return Double4(vmaxq_f64(a.lo, b.lo), vmaxq_f64(a.hi, b.hi));
}
PSYNTH_FORCE_INLINE Double4 min(const Double4& a, const Double4& b) {
  return Double4(vminq_f64(a.lo, b.lo), vminq_f64(a.hi, b.hi));
}
PSYNTH_FORCE_INLINE Double4 fmadd(const Double4& a, const Double4& b, const Double4& c) {
  return Double4(vfmaq_f64(c.lo, a.lo, b.lo), vfmaq_f64(c.hi, a.hi, b.hi));
}

#else  // Scalar fallback

struct Double4 {
  double v[4];

  PSYNTH_FORCE_INLINE Double4() : v{0, 0, 0, 0} {}
  PSYNTH_FORCE_INLINE Double4(double x) : v{x, x, x, x} {}
  PSYNTH_FORCE_INLINE Double4(double a, double b, double c, double d) : v{a, b, c, d} {}

  PSYNTH_FORCE_INLINE static Double4 load(const double* ptr) {
    return Double4(ptr[0], ptr[1], ptr[2], ptr[3]);
  }
  PSYNTH_FORCE_INLINE void store(double* ptr) const {
    ptr[0] = v[0]; ptr[1] = v[1]; ptr[2] = v[2]; ptr[3] = v[3];
  }

  PSYNTH_FORCE_INLINE Double4 operator+(const Double4& o) const {
    return Double4(v[0]+o.v[0], v[1]+o.v[1], v[2]+o.v[2], v[3]+o.v[3]);
  }
  PSYNTH_FORCE_INLINE Double4 operator-(const Double4& o) const {
    return Double4(v[0]-o.v[0], v[1]-o.v[1], v[2]-o.v[2], v[3]-o.v[3]);
  }
  PSYNTH_FORCE_INLINE Double4 operator*(const Double4& o) const {
    return Double4(v[0]*o.v[0], v[1]*o.v[1], v[2]*o.v[2], v[3]*o.v[3]);
  }
  PSYNTH_FORCE_INLINE Double4 operator/(const Double4& o) const {
    return Double4(v[0]/o.v[0], v[1]/o.v[1], v[2]/o.v[2], v[3]/o.v[3]);
  }

  PSYNTH_FORCE_INLINE double operator[](int i) const { return v[i]; }
  PSYNTH_FORCE_INLINE double horizontal_sum() const { return v[0]+v[1]+v[2]+v[3]; }
};

PSYNTH_FORCE_INLINE Double4 sqrt(const Double4& x) {
  return Double4(std::sqrt(x.v[0]), std::sqrt(x.v[1]), std::sqrt(x.v[2]), std::sqrt(x.v[3]));
}
PSYNTH_FORCE_INLINE Double4 max(const Double4& a, const Double4& b) {
  return Double4(std::max(a.v[0],b.v[0]), std::max(a.v[1],b.v[1]),
                 std::max(a.v[2],b.v[2]), std::max(a.v[3],b.v[3]));
}
PSYNTH_FORCE_INLINE Double4 min(const Double4& a, const Double4& b) {
  return Double4(std::min(a.v[0],b.v[0]), std::min(a.v[1],b.v[1]),
                 std::min(a.v[2],b.v[2]), std::min(a.v[3],b.v[3]));
}
PSYNTH_FORCE_INLINE Double4 fmadd(const Double4& a, const Double4& b, const Double4& c) {
  return a * b + c;
}

#endif

// ============================================================================
// Float8 - 8 packed floats (AVX or 2x SSE or scalar)
// ============================================================================

#if defined(PSYNTH_SIMD_AVX2) || defined(PSYNTH_SIMD_AVX512)

struct Float8 {
  __m256 v;

  PSYNTH_FORCE_INLINE Float8() : v(_mm256_setzero_ps()) {}
  PSYNTH_FORCE_INLINE Float8(__m256 x) : v(x) {}
  PSYNTH_FORCE_INLINE Float8(float x) : v(_mm256_set1_ps(x)) {}

  PSYNTH_FORCE_INLINE static Float8 load(const float* ptr) { return _mm256_loadu_ps(ptr); }
  PSYNTH_FORCE_INLINE static Float8 load_aligned(const float* ptr) { return _mm256_load_ps(ptr); }
  PSYNTH_FORCE_INLINE void store(float* ptr) const { _mm256_storeu_ps(ptr, v); }
  PSYNTH_FORCE_INLINE void store_aligned(float* ptr) const { _mm256_store_ps(ptr, v); }

  PSYNTH_FORCE_INLINE Float8 operator+(const Float8& o) const { return _mm256_add_ps(v, o.v); }
  PSYNTH_FORCE_INLINE Float8 operator-(const Float8& o) const { return _mm256_sub_ps(v, o.v); }
  PSYNTH_FORCE_INLINE Float8 operator*(const Float8& o) const { return _mm256_mul_ps(v, o.v); }
  PSYNTH_FORCE_INLINE Float8 operator/(const Float8& o) const { return _mm256_div_ps(v, o.v); }

  PSYNTH_FORCE_INLINE Float8 operator<(const Float8& o) const { return _mm256_cmp_ps(v, o.v, _CMP_LT_OQ); }
  PSYNTH_FORCE_INLINE Float8 operator&(const Float8& o) const { return _mm256_and_ps(v, o.v); }

  PSYNTH_FORCE_INLINE int movemask() const { return _mm256_movemask_ps(v); }

  PSYNTH_FORCE_INLINE float horizontal_sum() const {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
  }
};

PSYNTH_FORCE_INLINE Float8 sqrt(const Float8& x) { return _mm256_sqrt_ps(x.v); }

#elif defined(PSYNTH_SIMD_SSE42)

struct Float8 {
  __m128 lo, hi;

  PSYNTH_FORCE_INLINE Float8() : lo(_mm_setzero_ps()), hi(_mm_setzero_ps()) {}
  PSYNTH_FORCE_INLINE Float8(__m128 l, __m128 h) : lo(l), hi(h) {}
  PSYNTH_FORCE_INLINE Float8(float x) : lo(_mm_set1_ps(x)), hi(_mm_set1_ps(x)) {}

  PSYNTH_FORCE_INLINE static Float8 load(const float* ptr) {
    return Float8(_mm_loadu_ps(ptr), _mm_loadu_ps(ptr + 4));
  }
  PSYNTH_FORCE_INLINE void store(float* ptr) const {
    _mm_storeu_ps(ptr, lo);
    _mm_storeu_ps(ptr + 4, hi);
  }

  PSYNTH_FORCE_INLINE Float8 operator+(const Float8& o) const {
    return Float8(_mm_add_ps(lo, o.lo), _mm_add_ps(hi, o.hi));
  }
  PSYNTH_FORCE_INLINE Float8 operator*(const Float8& o) const {
    return Float8(_mm_mul_ps(lo, o.lo), _mm_mul_ps(hi, o.hi));
  }

  PSYNTH_FORCE_INLINE float horizontal_sum() const {
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
  }
};

#else  // Scalar

struct Float8 {
  float v[8];

  PSYNTH_FORCE_INLINE Float8() : v{0,0,0,0,0,0,0,0} {}
  PSYNTH_FORCE_INLINE Float8(float x) : v{x,x,x,x,x,x,x,x} {}

  PSYNTH_FORCE_INLINE static Float8 load(const float* ptr) {
    Float8 r;
    for(int i=0;i<8;i++) r.v[i] = ptr[i];
    return r;
  }
  PSYNTH_FORCE_INLINE void store(float* ptr) const {
    for(int i=0;i<8;i++) ptr[i] = v[i];
  }

  PSYNTH_FORCE_INLINE Float8 operator+(const Float8& o) const {
    Float8 r;
    for(int i=0;i<8;i++) r.v[i] = v[i] + o.v[i];
    return r;
  }
  PSYNTH_FORCE_INLINE Float8 operator*(const Float8& o) const {
    Float8 r;
    for(int i=0;i<8;i++) r.v[i] = v[i] * o.v[i];
    return r;
  }

  PSYNTH_FORCE_INLINE float horizontal_sum() const {
    return v[0]+v[1]+v[2]+v[3]+v[4]+v[5]+v[6]+v[7];
  }
};

#endif

// ============================================================================
// Batch processing utilities
// ============================================================================

// Process array in SIMD chunks with remainder handling
template<typename Func>
PSYNTH_FORCE_INLINE void process_batch_4(int N, Func&& func) {
  const int N4 = N & ~3;  // Round down to multiple of 4
  for (int i = 0; i < N4; i += 4) {
    func(i, 4);
  }
  if (N4 < N) {
    func(N4, N - N4);  // Handle remainder
  }
}

// Prefetch hint for upcoming data access
PSYNTH_FORCE_INLINE void prefetch_read(const void* ptr) {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(ptr, 0, 3);
#elif defined(_MSC_VER)
  _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0);
#endif
}

PSYNTH_FORCE_INLINE void prefetch_write(void* ptr) {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(ptr, 1, 3);
#elif defined(_MSC_VER)
  _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0);
#endif
}

}  // namespace psynth::simd
