using Flux
using Flux: update!
using Statistics
using DataFrames
using CSV
using Optimisers

##

begin # import from CSV
    df = DataFrame(CSV.File("A:/_Coding/Julia/polycycle/oil_prod.csv"))
    y = df.oil_prod
    y_n = y ./ maximum(y)
    x = 1:size(y)[1]
end

##

begin #hubbert model
    θ = ([1.0f0], [0.1f0], [maximum(x) / 2])
    predict(θ, x) = @. 2θ[1] / (1 + cosh(θ[2] * (x - θ[3])))
end

# Cost function(currently mean squared error)
J(θ, x, y) = Flux.Losses.mse(predict(θ, x), y)

##

# works (slowly) with Descent other optimisers cause error
# opt = Descent(0.01)
opt = Optimisers.setup(Optimisers.Momentum(), θ)

function train_iter(opt, θ, x, y)
    loss, dJdθ = Flux.withgradient(θ -> J(θ, x, y), θ)
    opt, θ = Optimisers.update!(opt, θ, dJdθ[1])
    return loss
end

##

# running this causes error when not using Descent
train_iter(opt, θ, x, y_n)