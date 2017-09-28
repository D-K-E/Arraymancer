## Reduction


proc bp_mean[T](input_shape: seq[int]): BackProp[T] =
  return proc(gradient: Tensor[T]): Tensor[T] =
    # Note: gradient isScalar
    let scal_grad = gradient / T(input_shape.foldl(a * b)) ## Note the backpropagation is wrong ...
    return scal_grad

proc mean*[T](v: Variable[T]): Variable[T] =
  let m = [v.value.mean].toTensor
  return Variable[T](
           tape: v.tape,
           value: m,
           index: v.tape.push_unary(v.index, bp_mean[T](v.value.shape))
  )