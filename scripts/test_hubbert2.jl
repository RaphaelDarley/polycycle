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
    # θ = [[1.0f0], [1.0f0], [1.0f0]]
    θ = [[1.0f0], [0.1f0], [maximum(x) / 2]]
    predict(θ, x) = @. 2θ[1] / (1 + cosh(θ[2] * (x - θ[3])))
end

plot((x) -> 0.25 * (sech(x / 2))^2)
plot((x) -> predict(θ, x)[1], 0, 100)


# params = Flux.params(θ)
J(θ, x, y) = Flux.Losses.mse(predict(θ, x), y)
# grads = gradient(J, θ)[1]

opt = Descent(0.01)
opt = Adam()
opt = Momentum()


function train_iter(θ, x, y)
    # dJdθ, _, _ = gradient(J, θ, x, y)
    dJdθ = gradient((θ) -> J(θ, x, y), θ)[1]
    update!(opt, θ, dJdθ)
    # println(θ)
    return J(θ, x, y_n)
end;

train_iter(θ, x, y_n);

cost_arr = [NaN]

for i = 1:100000
    # push!(cost_arr, train_iter(θ, x, y_n))
    # if (cost_arr[end-1] - cost_arr[end]) / cost_arr[end-1] < 0.0001
    #     break
    # end
end

function run_iters(n)
    for i = 1:n
        train_iter(θ, x, y_n)
    end
end;

for i = 1:100
    run_iters(10000)
    plot!((x) -> (hubbert_model(θ, x)[1]), minimum(x), maximum(x), label="Custom model", lw=2)
end


plot(1:size(cost_arr)[1], cost_arr)


begin
    # plot(reshape(x, (61, 1)), reshape(y, (61, 1)), lw=3, seriestype=:scatter, label="", title="Simple Linear Regression", xlabel="x", ylabel="y")
    scatter(x, y ./ maximum(y))
    plot!((x) -> (hubbert_model(θ, x)[1]), minimum(x), maximum(x), label="Custom model", lw=2)
end

plot!((x) -> (hubbert_model(θ, x)[1]), minimum(x), maximum(x), label="Custom model", lw=2)

plot!((x) -> (hubbert_model([[0.75f0], [0.08f0], [75.0f0]], x)[1]), minimum(x), maximum(x), label="Custom model", lw=2)