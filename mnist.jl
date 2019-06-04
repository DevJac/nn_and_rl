using Flux
using MLBase: correctrate
using MLDatasets: MNIST
using Printf: @printf

const digits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]

const model = Chain(
    Dense(28  * 28, 100, relu),
    Dense(100, 10, relu),
    softmax)

const x0, y0 = MNIST.traindata()
const x = reshape(x0, size(x0)[1] * size(x0)[2], size(x0)[3])
const y = Flux.onehotbatch(y0, digits)

loss(x, y) = Flux.crossentropy(model(x), y)

const optimizer = NADAM()

function train(epochs)
    @printf(
        "Epoch %4d / %4d - Loss: %8.4f    Correct: %8.2f%%\n",
        0,
        epochs,
        loss(x, y).data,
        correctrate(to_class(model(x)), to_class(y)) * 100)
    for epoch in 1:epochs
        Flux.train!(loss, params(model), [(x, y)], optimizer)
        @printf(
            "Epoch %4d / %4d - Loss: %8.4f    Correct: %8.2f%%\n",
            epoch,
            epochs,
            loss(x, y).data,
            correctrate(to_class(model(x)), to_class(y)) * 100)
    end
end

function to_class(a)
    reshape(
        mapslices(
            argmax,
            a,
            dims=1),
        size(a)[2])
end

to_digit(a) = map((i) -> digits[i], to_class(a))
