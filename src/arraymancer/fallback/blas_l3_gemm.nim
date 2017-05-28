# Copyright 2017 Mamy André-Ratsimbazafy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The following code is heavily inspired by ulmBLAS (http://apfel.mathematik.uni-ulm.de/~lehn/ulmBLAS/)
# which is heavily inspired by BLIS (https://github.com/flame/blis)
# A big difference (for now?) is instead of passing (const) pointers I pass the (var) array and a var offset.

## Reading
# C++ version: https://stackoverflow.com/questions/35620853/how-to-write-a-matrix-matrix-product-that-can-compete-with-eigen
# uBLAS C++: http://www.mathematik.uni-ulm.de/~lehn/test_ublas/session1/page01.html
# Blaze C++: http://www.mathematik.uni-ulm.de/~lehn/test_blaze/session1/page01.html
# Rust BLIS inspired: https://github.com/bluss/matrixmultiply

#### TODO:
# - OpenMP parallelization
# {.passl: "-fopenmp".} # Issue: Clang OSX does not support openmp
# {.passc: "-fopenmp".} # and the default GCC is actually a link to Clang

# - Loop unrolling  # Currently Nim `unroll` pragma exists but is ignored.
# - Pass `-march=native` to the compiler
# - Align memory # should be automatic
# - Is there a way to get L1/L2 cache size at compile-time
# - Is there a way to get number of registers at compile-time

# Best numbers depend on
# L1, L2, L3 cache and register size
const MC = 384
const KC = 384
const NC = 4096

const MR = 4
const NR = 4

const MCKC = MC*KC
const KCNC = KC*NC
const MRNR = MR*NR

include ./blas_l3_gemm_data_structure.nim
include ./blas_l3_gemm_packing
include ./blas_l3_gemm_aux
include ./blas_l3_gemm_micro_kernel
include ./blas_l3_gemm_macro_kernel

# We use T: int so that it is easy to change to float to benchmark against OpenBLAS/MKL/BLIS
proc gemm_nn[T](m, n, k: int,
                alpha: T,
                A: seq[T], offA: int,
                incRowA, incColA: int,
                B: seq[T], offB: int,
                incRowB, incColB: int,
                beta: T,
                C: var seq[T], offC: int,
                incRowC, incColC: int) = # {.noSideEffect.} =

  let
    mb = (m + MC - 1) div MC
    nb = (n + NC - 1) div NC
    kb = (k + KC - 1) div KC

    mod_mc = m mod MC
    mod_nc = n mod NC
    mod_kc = k mod KC

  var mc, nc, kc: int
  var tmp_beta: T

  var (buffer_A, pbuf_A) = newBufferArrayPtr(MCKC, T)
  var (buffer_B, pbuf_B) = newBufferArrayPtr(KCNC, T)
  var (buffer_C, pbuf_C) = newBufferArrayPtr(MRNR, T)

  var pA = A.to_ptr + offA
  var pB = B.to_ptr + offB
  var pC = C.to_ptr + offC

  if alpha == 0.T or k == 0:
    gescal(m, n, beta, pC, incRowC, incColC)
    return

  for j in 0 ..< nb:
    nc =  if (j != nb-1 or mod_nc == 0): NC
          else: mod_nc

    for k in 0 ..< kb:
      kc       =  if (k != kb-1 or mod_kc == 0): KC
                  else: mod_kc
      tmp_beta =  if k == 0: beta
                  else: 1.T
      pack_dim( nc, kc,
                pB + k*KC*incRowB + j*NC*incColB,
                incColB, incRowB, NR,
                pbuf_B)
      for i in 0 ..< mb:
        mc = if (i != mb-1 or mod_mc == 0): MC
             else: mod_mc

        pack_dim( mc, kc,
                  pA + i*MC*incRowA+k*KC*incColA,
                  incRowA, incColA, MR,
                  pbuf_A)

        gemm_macro_kernel(mc, nc, kc,
                          alpha, tmp_beta,
                          pC + i*MC*incRowC + j*NC*incColC,
                          incRowC, incColC, pbuf_A, pbuf_B, pbuf_C)