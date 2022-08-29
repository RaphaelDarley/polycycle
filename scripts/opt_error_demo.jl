using Flux
using Flux: update!
using Plots
using Statistics
using DataFrames
using CSV

begin # import from CSV
    df = DataFrame(CSV.File("A:/_Coding/Julia/polycycle/oil_prod.csv"))
    y = df.oil_prod
    y_n = y ./ maximum(y)
    x = 1:size(y)[1]
end

begin #hubbert model
    θ = [[1.0f0], [0.1f0], [maximum(x) / 2]]
    predict(θ, x) = @. 2θ[1] / (1 + cosh(θ[2] * (x - θ[3])))
end

# Cost function(currently mean squared error)
J(θ, x, y) = Flux.Losses.mse(predict(θ, x), y)


# works (slowly) with Descent other optimisers cause error
opt = Descent(0.01)
opt = Adam()
opt = Momentum()


function train_iter(θ, x, y)
    dJdθ = gradient((θ) -> J(θ, x, y), θ)[1]
    update!(opt, θ, dJdθ)
    return J(θ, x, y)
end;

# running this causes error when not using Descent
train_iter(θ, x, y_n);

begin # graph data with current curve
    scatter(x, y ./ maximum(y))
    plot!((x) -> (hubbert_model(θ, x)[1]), minimum(x), maximum(x), label="Custom model", lw=2)
end
