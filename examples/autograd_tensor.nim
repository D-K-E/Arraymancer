import ../src/arraymancer, ../src/arraymancer_ag


let ctx = newContext[float32]()


let
    a = ctx.variable(randomTensor(10,10,100).astype(float32))
    b = ctx.variable(randomTensor(10,10,100).astype(float32))
    c = ctx.variable(randomTensor(10,10,100).astype(float32))
    x = ctx.variable(randomTensor(10,10,100).astype(float32))
    y = ctx.variable(randomTensor(10,10,100).astype(float32))


proc forwardNeuron[T](a,b,c,x,y: T): T =
    let
        ax = a * x
        by = b * y
        axpby = ax + by
        axpbypc = axpby + c
        # s = axpbypc.sigmoid()
    return axpbypc


var s = forwardNeuron(a,b,c,x,y)


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