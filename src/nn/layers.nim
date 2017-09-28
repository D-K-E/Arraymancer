# Copyright 2017 The Arraymancer contributors
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

type Initialization* = enum
  kaiming, glorot

type Layer*[T] = object {.inheritable.}
  ## Generic layer object that each other layer inherits.

type Linear* {.final.} [T] = ref object of Layer[T]
  ## Linear transformation layer. Also known as "Dense" or "Fully-Connected"
  ## Does y = Ax + b
  weight: Tensor[T]
  bias: Tensor[T]
  cache: Tensor[T] ## Cache the input for bprop, will not be stored here when autograd is done

proc newLinear*[T](shape: varargs[int], init: Initialization = kaiming): Linear[T] =
  new result
  result.weight = newTensor[T](shape)
  result.bias = newTensor[T](shape)

method forward*[T](self: Layer, t: Tensor[T]): Tensor[T] {.base.}=
  raise newException(ValueError, "Forward method not implemented")

method forward*[T](self: Linear[T], t: Tensor[T]): Tensor[T] {.base.}=
  self.cache = t
  return self.weight * t + self.bias ## Redo the CPU BLAS for fused ops
