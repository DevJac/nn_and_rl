using Flux
using MLDatasets: MNIST

const digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

model = Chain(
    Dense(28  * 28, 100, relu),
    Dense(100, 10, relu),
    softmax)

x0, y0 = MNIST.traindata()
x = reshape(x0, size(x0)[1] * size(x0)[2], size(x0)[3])
y = Flux.onehotbatch(y0, digits)

loss(x, y) = Flux.crossentropy(model(x), y)

optimizer = NADAM()

function train()
    println(loss(x, y))
    for _ in 1:20
        Flux.train!(loss, params(model), [(x, y)], optimizer)
        println(loss(x, y))
    end
end
