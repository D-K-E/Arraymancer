type
  # T is float32 or float64, for memory/accuracy tradeoff
  # We do not use Node[T: SomeReal] so that T can be nil in proc newContext[T]

  # To ease search, backward propagation procedures are prefixed with bp_
  BackProp[T] = proc (gradient: Tensor[T]): Tensor[T]

  Node[T] = object
    ## Represent an operation
    ## Stores the gradient transformation for backprop in weights
    ## Stores indices of parent operation in parents
    weights: array[2, BackProp[T]]
    parents: array[2,int] #ref indices to parent nodes

  Context*[T] = ref object
    ## Tape / Wengert list. Contains the list of applied operations
    nodes: seq[Node[T]]

  Variable*[T] = object
    ## Wrapper for values
    tape: Context[T]
    index: int
    value: Tensor[T]

  Grad[T] = object
    ## Wrapper for the list of gradients with regards to each inputs
    derivs: ref seq[Tensor[T]]

# Templates in Nim are always inlined. They are used for performance reason to save on function calls costs.

proc newContext*[T]: Context[T] {.noSideEffect.} =
  ## Initialize a context (Tape / Wengert list)
  new result
  result.nodes = newSeq[Node[T]]()

template len[T](t: Context[T]): int =
  ## Returns the number of operations applied in the context
  t.nodes.len()

template push[T](t: Context[T], node: Node[T]) =
  ## Append a new operation to the context
  t.nodes.add(node) #Appending in Nim is add not push

proc push_nullary[T](t: Context[T]): int {.noSideEffect.} =
  ## Append a nullary operation to the context
  let len = t.len()
  proc bp_0[T](gradient: Tensor[T]): Tensor[T] {.noSideEffect, closure.}= zeros[T](1,1)

  t.push(
    Node[T](
      weights: [bp_0[T], bp_0[T]],
      parents: [len, len]
      )
    )
  return len

proc push_unary[T](t: Context[T], parent0: int, weight0: BackProp[T]): int {.noSideEffect.} =
  ## Append a unary operation to the context
  let len = t.len()
  proc bp_0[T](gradient: Tensor[T]): Tensor[T] {.noSideEffect, closure.}= zeros[T](1,1)

  t.push(
    Node[T](
      weights: [weight0, bp_0[T]],
      parents: [parent0, len]
      )
    )
  return len

proc push_binary[T](t: Context[T], parent0: int, weight0: BackProp[T], parent1: int, weight1: BackProp[T]): int {.noSideEffect.} =
  ## Append a binary operation to the context
  let len = t.len()
  t.push(
    Node[T](
      weights: [weight0, weight1],
      parents: [parent0, parent1]
      )
    )
  return len

proc variable*[T](t: Context[T], value: Tensor[T]): Variable[T] {.noSideEffect.} =
  ## Wrap a variable to the context
  return Variable[T](
           tape: t,
           value: value,
           index: t.push_nullary()
           )

template value*[T](v: Variable[T]): Tensor[T]  =
  ## Unwrap the value from its context
  v.value

proc grad*[T](v: Variable[T]): Grad[T] =
  ## Compute the gradients
  # Computation is done with gradient set to 1 for the final output value
  # If needed it can be set to an arbitrary value (e.g. -1)
  let len = v.tape.len()
  let nodes = v.tape.nodes
  # echo repr nodes # Check representation on stack/heap

  result.derivs = new seq[Tensor[T]]

  var derivs = newSeqWith(len, zeros[T](1,1))

  derivs[v.index] = ones[T](1,1) #by default 1 Tensor

  for i in countdown(len-1,0):
    let node = nodes[i]
    let deriv = derivs[i]

    for j in 0..1:
      derivs[node.parents[j]] = derivs[node.parents[j]] .+ node.weights[j](deriv) # For broadcasting the 1 size tensors on both hand side

  result.derivs[] = derivs

template wrt*[T](g: Grad[T], v: Variable[T]): Tensor[T] =
  ## Get the gradient with regards to a specific input value
  g.derivs[v.index]


proc isScalar*[T](t: Tensor[T]): bool =
  for dim in t.shape:
    if dim != 1 and dim != 0:
      return false
  return true