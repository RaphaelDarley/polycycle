using Flux

W = rand(2, 5)
b = rand(2)

predict(x) = (W * x) .+ b
loss(x, y) = sum((predict(x) .- y) .^ 2)

x, y = rand(5), rand(2) # Dummy data
l = loss(x, y) # ~ 3

θ = Flux.params(W, b)
grads = gradient(() -> loss(x, y), θ)

predict(x)

using Flux: update!

opt = Descent(0.1) # Gradient descent with learning rate 0.1

for p in (W, b)
    update!(opt, p, grads[p])
end