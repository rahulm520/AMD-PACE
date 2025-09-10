/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#include <immintrin.h>
#include <ops/kernels/binary_kernels.h>

namespace pace {

namespace kernels {

namespace impl {

// Intrinsic definition : (AxB) + C for inputs in order
// FLOAT, INT8, INT8 -> output (INT8)
template <typename Taddend>
void qmul_add_kernel_mx96(
    float* a,
    uint8_t* b,
    float b_scale,
    int b_zpoint,
    Taddend* c,
    float c_scale,
    int c_zpoint,
    uint8_t* output,
    float o_scale,
    int o_zpoint,
    int M,
    int N) {
  int nblocks = 96;

  __m512 a1_float, a2_float, a3_float, a4_float, a5_float, a6_float;
  __m512 b1_float, b2_float, b3_float, b4_float, b5_float, b6_float;
  __m512 c1_float, c2_float, c3_float, c4_float, c5_float, c6_float;
  __m512 o1_float, o2_float, o3_float, o4_float, o5_float, o6_float;

  int i, j;

  int rs = N;

  __m512i b_reg_zpoint = _mm512_set1_epi32(b_zpoint);
  __m512 b_reg_scale = _mm512_set1_ps(b_scale);

  __m512i c_reg_zpoint = _mm512_set1_epi32(c_zpoint);
  __m512 c_reg_scale = _mm512_set1_ps(c_scale);

  __m512i o_reg_zpoint = _mm512_set1_epi32(o_zpoint);
  __m512 o_reg_scale = _mm512_set1_ps(o_scale);

  int addr;

  __m512i temp1 = _mm512_set1_epi32(255);
  __m512i temp2 = _mm512_set1_epi32(0);

  for (i = 0; (i) < M; i += 1) {
    for (j = 0; (j) < N; j += nblocks) {
      addr = (i * rs) + (j);

      // load A in 6 registers in float
      a1_float = _mm512_load_ps(a + addr);
      a2_float = _mm512_load_ps(a + addr + (16));
      a3_float = _mm512_load_ps(a + addr + (32));
      a4_float = _mm512_load_ps(a + addr + (48));
      a5_float = _mm512_load_ps(a + addr + (64));
      a6_float = _mm512_load_ps(a + addr + (80));

      // load B in int8 , convert to s32, substarct zeropoint and convert it to
      // float - 6 registers
      b1_float = _mm512_cvtepi32_ps(_mm512_sub_epi32(
          _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i const*)(b + addr))),
          b_reg_zpoint));
      b2_float = _mm512_cvtepi32_ps(_mm512_sub_epi32(
          _mm512_cvtepu8_epi32(
              _mm_loadu_si128((__m128i const*)(b + addr + (16)))),
          b_reg_zpoint));
      b3_float = _mm512_cvtepi32_ps(_mm512_sub_epi32(
          _mm512_cvtepu8_epi32(
              _mm_loadu_si128((__m128i const*)(b + addr + (32)))),
          b_reg_zpoint));
      b4_float = _mm512_cvtepi32_ps(_mm512_sub_epi32(
          _mm512_cvtepu8_epi32(
              _mm_loadu_si128((__m128i const*)(b + addr + (48)))),
          b_reg_zpoint));
      b5_float = _mm512_cvtepi32_ps(_mm512_sub_epi32(
          _mm512_cvtepu8_epi32(
              _mm_loadu_si128((__m128i const*)(b + addr + (64)))),
          b_reg_zpoint));
      b6_float = _mm512_cvtepi32_ps(_mm512_sub_epi32(
          _mm512_cvtepu8_epi32(
              _mm_loadu_si128((__m128i const*)(b + addr + (80)))),
          b_reg_zpoint));

      // Deqnt B values
      b1_float = _mm512_mul_ps(b1_float, b_reg_scale);
      b2_float = _mm512_mul_ps(b2_float, b_reg_scale);
      b3_float = _mm512_mul_ps(b3_float, b_reg_scale);
      b4_float = _mm512_mul_ps(b4_float, b_reg_scale);
      b5_float = _mm512_mul_ps(b5_float, b_reg_scale);
      b6_float = _mm512_mul_ps(b6_float, b_reg_scale);

      // load C in int8 , convert to s32, substarct zeropoint and convert it to
      // float - 6 registers
      if (std::is_same<Taddend, float>::value) {
        c1_float = _mm512_load_ps(c + addr);
        c2_float = _mm512_load_ps(c + addr + (16));
        c3_float = _mm512_load_ps(c + addr + (32));
        c4_float = _mm512_load_ps(c + addr + (48));
        c5_float = _mm512_load_ps(c + addr + (64));
        c6_float = _mm512_load_ps(c + addr + (80));
      } else {
        c1_float = _mm512_cvtepi32_ps(_mm512_sub_epi32(
            _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i const*)(c + addr))),
            c_reg_zpoint));
        c2_float = _mm512_cvtepi32_ps(_mm512_sub_epi32(
            _mm512_cvtepu8_epi32(
                _mm_loadu_si128((__m128i const*)(c + addr + (16)))),
            c_reg_zpoint));
        c3_float = _mm512_cvtepi32_ps(_mm512_sub_epi32(
            _mm512_cvtepu8_epi32(
                _mm_loadu_si128((__m128i const*)(c + addr + (32)))),
            c_reg_zpoint));
        c4_float = _mm512_cvtepi32_ps(_mm512_sub_epi32(
            _mm512_cvtepu8_epi32(
                _mm_loadu_si128((__m128i const*)(c + addr + (48)))),
            c_reg_zpoint));
        c5_float = _mm512_cvtepi32_ps(_mm512_sub_epi32(
            _mm512_cvtepu8_epi32(
                _mm_loadu_si128((__m128i const*)(c + addr + (64)))),
            c_reg_zpoint));
        c6_float = _mm512_cvtepi32_ps(_mm512_sub_epi32(
            _mm512_cvtepu8_epi32(
                _mm_loadu_si128((__m128i const*)(c + addr + (80)))),
            c_reg_zpoint));

        // Deqnt C values
        c1_float = _mm512_mul_ps(c1_float, c_reg_scale);
        c2_float = _mm512_mul_ps(c2_float, c_reg_scale);
        c3_float = _mm512_mul_ps(c3_float, c_reg_scale);
        c4_float = _mm512_mul_ps(c4_float, c_reg_scale);
        c5_float = _mm512_mul_ps(c5_float, c_reg_scale);
        c6_float = _mm512_mul_ps(c6_float, c_reg_scale);
      }

      // FMA = (A x B) + C
      o1_float = _mm512_fmadd_ps(a1_float, b1_float, c1_float);
      o2_float = _mm512_fmadd_ps(a2_float, b2_float, c2_float);
      o3_float = _mm512_fmadd_ps(a3_float, b3_float, c3_float);
      o4_float = _mm512_fmadd_ps(a4_float, b4_float, c4_float);
      o5_float = _mm512_fmadd_ps(a5_float, b5_float, c5_float);
      o6_float = _mm512_fmadd_ps(a6_float, b6_float, c6_float);

      // Qnt C values without zero point
      o1_float = _mm512_mul_round_ps(
          o1_float,
          o_reg_scale,
          (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      o2_float = _mm512_mul_round_ps(
          o2_float,
          o_reg_scale,
          (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      o3_float = _mm512_mul_round_ps(
          o3_float,
          o_reg_scale,
          (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      o4_float = _mm512_mul_round_ps(
          o4_float,
          o_reg_scale,
          (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      o5_float = _mm512_mul_round_ps(
          o5_float,
          o_reg_scale,
          (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      o6_float = _mm512_mul_round_ps(
          o6_float,
          o_reg_scale,
          (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

      // add zeropoint and store C values from float to INT8
      _mm_stream_si128(
          (__m128i*)((uint8_t*)output + addr),
          _mm512_cvtepi32_epi8(_mm512_max_epi32(
              _mm512_min_epi32(
                  _mm512_add_epi32(_mm512_cvtps_epi32(o1_float), o_reg_zpoint),
                  temp1),
              temp2)));

      _mm_stream_si128(
          (__m128i*)((uint8_t*)output + addr + (16)),
          _mm512_cvtepi32_epi8(_mm512_max_epi32(
              _mm512_min_epi32(
                  _mm512_add_epi32(_mm512_cvtps_epi32(o2_float), o_reg_zpoint),
                  temp1),
              temp2)));

      _mm_stream_si128(
          (__m128i*)((uint8_t*)output + addr + (32)),
          _mm512_cvtepi32_epi8(_mm512_max_epi32(
              _mm512_min_epi32(
                  _mm512_add_epi32(_mm512_cvtps_epi32(o3_float), o_reg_zpoint),
                  temp1),
              temp2)));

      _mm_stream_si128(
          (__m128i*)((uint8_t*)output + addr + (48)),
          _mm512_cvtepi32_epi8(_mm512_max_epi32(
              _mm512_min_epi32(
                  _mm512_add_epi32(_mm512_cvtps_epi32(o4_float), o_reg_zpoint),
                  temp1),
              temp2)));

      _mm_stream_si128(
          (__m128i*)((uint8_t*)output + addr + (64)),
          _mm512_cvtepi32_epi8(_mm512_max_epi32(
              _mm512_min_epi32(
                  _mm512_add_epi32(_mm512_cvtps_epi32(o5_float), o_reg_zpoint),
                  temp1),
              temp2)));

      _mm_stream_si128(
          (__m128i*)((uint8_t*)output + addr + (80)),
          _mm512_cvtepi32_epi8(_mm512_max_epi32(
              _mm512_min_epi32(
                  _mm512_add_epi32(_mm512_cvtps_epi32(o6_float), o_reg_zpoint),
                  temp1),
              temp2)));
    }
  }

  return;
}

template void qmul_add_kernel_mx96<float>(
    float*,
    uint8_t*,
    float,
    int,
    float*,
    float,
    int,
    uint8_t*,
    float,
    int,
    int,
    int);
template void qmul_add_kernel_mx96<uint8_t>(
    float*,
    uint8_t*,
    float,
    int,
    uint8_t*,
    float,
    int,
    uint8_t*,
    float,
    int,
    int,
    int);

} // namespace impl

} // namespace kernels
} // namespace pace
