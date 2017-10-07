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

type
  Backend* = enum
    ## ``Backend`` for tensor computation and memory allocation.
    ##
    ##
    ## Only deprecated procs from v0.1.3 uses this for the moment.
    Cpu,
    Cuda

  MetadataArray = array[MAXRANK, int]

  Tensor*[T] = object
    # Size of the datastructure is 32 bytes - perfect !
    ## Tensor data structure stored on Cpu
    ##   - ``shape``: Dimensions of the tensor
    ##   - ``strides``: Numbers of items to skip to get the next item along a dimension.
    ##   - ``offset``: Offset to get the first item of the Tensor. Note: offset can be negative, in particular for slices.
    ##   - ``data``: A sequence that holds the actual data
    ## Fields are public so that external libraries can easily construct a Tensor.
    rank*: int
    shape*: MetadataArray
    strides*: MetadataArray
    offset*: int
    data*: seq[T] # Perf note: seq are always deep copied on "var" assignement.

  CudaSeq* [T: SomeReal] = object
    ## Seq-like structure on the Cuda backend.
    ##
    ## Nim garbage collector will automatically ask cuda to clear GPU memory if ``data`` becomes unused.
    len: int
    data: ref[ptr UncheckedArray[T]]

  CudaTensor*[T: SomeReal] = object
    ## Tensor data structure stored on Nvidia GPU (Cuda)
    ##   - ``shape``: Dimensions of the tensor
    ##   - ``strides``: Numbers of items to skip to get the next item along a dimension.
    ##   - ``offset``: Offset to get the first item of the Tensor. Note: offset can be negative, in particular for slices.
    ##   - ``data``: A cuda seq-like object that points to the data location
    ## Note: currently ``=`` assignement for CudaTensor does not copy. Both CudaTensors will share a view of the same data location.
    ## Modifying the data in one will modify the data in the other.
    ##
    ## In the future CudaTensor will leverage Nim compiler to automatically
    ## copy if a memory location would be used more than once in a mutable manner.
    rank*: int
    shape*: seq[int]
    strides*: seq[int]
    offset*: int
    data*: CudaSeq[T] # Memory on Cuda device will be automatically garbage-collected

  AnyTensor*[T] = Tensor[T] or CudaTensor[T]

proc toMetadataArray*(va: varargs[int]): MetadataArray {.inline,noSideEffect.}=
  for i in 0 ..< va.len:
    result[i] = va[i]

proc toSeq*(mda: MetadataArray): seq[int] {.inline,noSideEffect.} =
  result = @[]
  var i = 0
  while i < mda.len and mda[i] != 0:
    result.add(mda[i])

proc size*(t: AnyTensor): int {.noSideEffect, inline.}=
  ## Input:
  ##     - A tensor
  ## Returns:
  ##     - The total number of elements it contains
  result = 1
  for i in 0 ..< t.rank:
    result *= t.shape[i]

proc shape_to_strides*(shape: seq[int], layout: OrderType = rowMajor): seq[int] {.deprecated, noSideEffect.} =
  ## DEPRECATED: Arraymancer does not used seq anymore to store tensor metadata
  ##
  ## Input:
  ##     - A shape (seq of int), for example @[3,5] for a 3x5 matrix
  ##     - Optionally rowMajor (C layout - default) or colMajor (Fortran)
  ## Returns:
  ##     - The strides in C or Fortran order corresponding to this shape and layout
  ##
  ## Arraymancer defaults to rowMajor. Temporarily, CudaTensors are colMajor by default.
  # See Design document for further considerations.
  if layout == rowMajor:
    return (shape & 1)[1..shape.len].scanr(a * b)

  return (1 & shape)[0..shape.high].scanl(a * b)

proc mstrides_from_shape(strides: var MetadataArray,
                          shape: MetadataArray,
                          rank: int,
                          layout: OrderType = rowMajor) {.noSideEffect.} =
  ## Input:
  ##     - Strides that will be modified in place
  ##     - Rank of the tensor
  ##     - Optionally rowMajor (C layout - default) or colMajor (Fortran)
  ##
  ## Arraymancer defaults to rowMajor. Temporarily, CudaTensors are colMajor by default.
  # See Design document for further considerations.

  # TODO: explore returning "accum" as lots of initialization do a loop
  # to get tensor.size

  var accum = 1

  if layout == rowMajor:
    for i in countdown(rank-1,0):
      strides[i] = accum
      accum *= shape[i]
    return

  for i in 0 ..< rank:
    strides[i] = accum
    accum *= shape[i]
  return

proc is_C_contiguous*(t: AnyTensor): bool {.noSideEffect,inline.}=
  ## Check if the tensor follows C convention / is row major
  var z = 1
  for i in countdown(t.shape.high,0):
    # 1. We should ignore strides on dimensions of size 1
    # 2. Strides always must have the size equal to the product of the next dimensons
    if t.shape[i] != 1 and t.strides[i] != z:
        return false
    z *= t.shape[i]
  return true

proc is_F_contiguous*(t: AnyTensor): bool {.noSideEffect,inline.}=
  ## Check if the tensor follows Fortran convention / is column major
  var z = 1
  for i in 0..<t.shape.len:
    # 1. We should ignore strides on dimensions of size 1
    # 2. Strides always must have the size equal to the product of the next dimensons
    if t.shape[i] != 1 and t.strides[i] != z:
        return false
    z *= t.shape[i]
  return true

proc isContiguous*(t: AnyTensor): bool {.noSideEffect,inline.}=
  ## Check if the tensor is contiguous
  return t.is_C_contiguous or t.is_F_contiguous

template get_data_ptr*[T](t: AnyTensor[T]): ptr T =
  ## Input:
  ##     - A tensor
  ## Returns:
  ##     - A pointer to the start of its data
  when t is Tensor:
    unsafeAddr(t.data[0])
  elif t is CudaTensor:
    unsafeAddr(t.data.data[0])

proc dataArray*[T](t: Tensor[T]): ptr UncheckedArray[T] {.inline.} =
  cast[ptr UncheckedArray[T]](t.data[t.offset].unsafeAddr)
