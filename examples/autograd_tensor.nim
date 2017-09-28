import ../src/arraymancer, ../src/arraymancer_ag, sequtils
# This example is false, however it was proof enough that working with closure
# For a tensor autograd is a huge pain

let ctx = newContext[float32]()


let
    a = ctx.variable(toSeq(1..12).toTensor.reshape(3,4).astype(float32))
    b = ctx.variable(toSeq(2..13).toTensor.reshape(3,4).astype(float32))
    c = ctx.variable(toSeq(3..11).toTensor.reshape(3,3).astype(float32))
    x = ctx.variable(toSeq(4..15).toTensor.reshape(4,3).astype(float32))
    y = ctx.variable(toSeq(5..16).toTensor.reshape(4,3).astype(float32))


proc forwardNeuron[T](a,b,c,x,y: T): T =
    let
        ax = a * x
        by = b * y
        axpby = ax + by
        axpbypc = axpby + c
        # s = axpbypc.sigmoid()
    return axpbypc


var s = mean forwardNeuron(a,b,c,x,y)


echo s.value


let gradient = s.grad()

echo "grad wrt a"
echo gradient.wrt(a)
echo "\ngrad wrt b"
echo gradient.wrt(b)
echo "\ngrad wrt c"
echo gradient.wrt(c)
echo "\ngrad wrt x"
echo gradient.wrt(x)
echo "\ngrad wrt y"
echo gradient.wrt(y)
