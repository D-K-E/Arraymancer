## Basic gradient transformations for backward pass
proc bp_identity[T](gradient: Tensor[T]): Tensor[T]= gradient
proc bp_negate[T](gradient: Tensor[T]): Tensor[T]= -gradient

proc `+`*[T](lhs: Variable[T], rhs: Variable[T]): Variable[T] =
  return Variable[T](
           tape: lhs.tape,
           value: lhs.value + rhs.value,
           index: lhs.tape.push_binary(
             lhs.index, bp_identity[T],
             rhs.index, bp_identity[T]
             )
           )

proc `-`*[T](lhs: Variable[T], rhs: Variable[T]): Variable[T] =
  return Variable[T](
           tape: lhs.tape,
           value: lhs.value - rhs.value,
           index: lhs.tape.push_binary(
             lhs.index, bp_identity[T],
             rhs.index, bp_negate[T]
             )
           )

# Multiplication
proc bp_mul[T](hs: Tensor[T]): BackProp[T] =
  return proc(gradient: Tensor[T]): Tensor[T] =
    if not gradient.isScalar:
      gradient * transpose(hs)
    else:
      gradient.data[0] * hs

proc `*`*[T](lhs: Variable[T], rhs: Variable[T]): Variable[T] =
  return Variable[T](
           tape: lhs.tape,
           value: lhs.value * rhs.value,
           index: lhs.tape.push_binary(
             lhs.index, bp_mul[T](rhs.value),
             rhs.index, bp_mul[T](lhs.value)
             )
  )