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


proc isNaiveIterable(t: AnyTensor): bool {.inline.}=
  ## If t is not a slice we can iterate with a naive for loop
  return t.data.len == t.size

proc isNaiveIterableWith(t1: AnyTensor, t2: AnyTensor): bool {.inline.}=
  ## If shape and strides are the same, we can iterate on both tensors at the same time
  ## modulo their offsets
  ## We don't need those to have data.len == size
  return (t1.strides == t2.strides) and (t1.shape == t2.shape)

proc getTransposeTarget(t: AnyTensor): TransposeType {.noSideEffect.}=
  ## TransposeType is introduced by ``nimblas``
  ## Default layout is Row major.
  ## Everytime it is worth it or fused with a BLAS operation we change the strides to Row-Major
  if is_C_contiguous(t): return TransposeType.noTranspose
  elif is_F_contiguous(t): return TransposeType.transpose
  else: raise newException(ValueError,"Operation not supported for this matrix. It has a non-contiguous layout")