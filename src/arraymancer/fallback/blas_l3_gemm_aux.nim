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

## Compute Y += alpha * X
proc geaxpy[T]( m, n: int,
                alpha: T,
                X: BufferPtr[MRNR, T],
                incRowX, incColX: int,
                pY: SeqPtr[T],
                incRowY, incColY: int) =

  var Y = pY

  if alpha != 1.T:
    for j in 0 ..< n:
      for i in 0 ..< m:
        Y[i*incRowY + j*incColY] += alpha * X[i*incRowX + j*incColX]
  else:
    for j in 0 ..< n:
      for i in 0 ..< m:
        Y[i*incRowY + j*incColY] += X[i*incRowX + j*incColX]

## Compute X *= alpha
proc gescal[T]( m, n: int,
                alpha: T,
                pX: SeqPtr[T],
                incRowX, incColX: int) =
  var X = pX

  if alpha != 0.T:
    for j in 0 ..< n:
      for i in 0 ..< m:
        X[i*incRowX + j*incColX] *= alpha
  else:
    for j in 0 ..< n:
      for i in 0 ..< m:
        X[i*incRowX + j*incColX] = 0