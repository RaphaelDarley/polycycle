using Flux
using Flux: update!
using Plots
using Statistics
using DataFrames
using CSV
using Optimisers

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

opt = Optimisers.setup(Optimisers.Momentum(), θ)
opt = Optimisers.setup(Optimisers.Adam(), θ)
opt = Optimisers.setup(Optimisers.Descent(), θ)

function train_iter(opt, θ, x, y)
    loss, dJdθ = Flux.withgradient(θ -> J(θ, x, y), θ)
    opt, θ = Optimisers.update!(opt, θ, dJdθ[1])
end

train_iter(opt, θ, x, y_n);

cost_arr = [NaN]

for i = 1:100000
    # push!(cost_arr, train_iter(θ, x, y_n))
    # if (cost_arr[end-1] - cost_arr[end]) / cost_arr[end-1] < 0.0001
    #     break
    # end
end

function run_iters(n)
    for i = 1:n
        train_iter(opt, θ, x, y_n)
    end
end;

for i = 1:10
    run_iters(1000)
    println(i, " loops done")
    push!(cost_arr, J(θ, x, y))
    # plot!((x) -> (predict(θ, x)[1]), minimum(x), maximum(x), label="Custom model", lw=2)
end

@time run_iters(100000)


plot(1:size(cost_arr)[1], cost_arr, label="momentum, base")
plot!(1:size(cost_arr)[1], cost_arr, label="Adam, base")
plot!(1:size(cost_arr)[1], cost_arr, label="Descent, base")

begin
    # plot(reshape(x, (61, 1)), reshape(y, (61, 1)), lw=3, seriestype=:scatter, label="", title="Simple Linear Regression", xlabel="x", ylabel="y")
    scatter(x, y ./ maximum(y))
    plot!((x) -> (predict(θ, x)[1]), minimum(x), maximum(x), label="Custom model", lw=2)
end



plot!((x) -> (predict(θ, x)[1]), minimum(x), maximum(x), label="Custom model", lw=2)

plot!((x) -> (predict([[0.75f0], [0.08f0], [75.0f0]], x)[1]), minimum(x), maximum(x), label="Custom model", lw=2)


optimiser_arr = [Optimisers.Descent, Optimisers.Momentum, Optimisers.Nesterov, Optimisers.RMSProp, Optimisers.Adam, Optimisers.RAdam, Optimisers.AdaMax, Optimisers.AdaGrad, Optimisers.AdaDelta, Optimisers.AMSGrad, Optimisers.NAdam, Optimisers.AdamW, Optimisers.OAdam, Optimisers.AdaBelief,]
optimiser_arr = [Optimisers.Momentum, Optimisers.Nesterov, Optimisers.RMSProp, Optimisers.Adam, Optimisers.RAdam, Optimisers.AdaMax, Optimisers.AdaGrad, Optimisers.AdaDelta, Optimisers.AMSGrad, Optimisers.NAdam, Optimisers.AdamW, Optimisers.OAdam, Optimisers.AdaBelief,]
optimiser_arr = [Optimisers.Adam, Optimisers.AdaMax, Optimisers.AdamW, Optimisers.OAdam,]
optimiser_arr = [Optimisers.AdaMax, Optimisers.OAdam]

plot(size=(1200, 1200))
scatter(x, y ./ maximum(y))
for op_fun = optimiser_arr
    println(String(Symbol(op_fun)))
    θ = [[1.0f0], [0.1f0], [maximum(x) / 2]]
    opt = Optimisers.setup(op_fun(), θ)
    cost_arr = [NaN]
    for i = 1:100
        run_iters(1000)
        push!(cost_arr, J(θ, x, y))
    end
    # plot!(1:size(cost_arr)[1], cost_arr, label=String(Symbol(op_fun)))
    plot!((x) -> (predict(θ, x)[1]), minimum(x), maximum(x), label=String(Symbol(op_fun)), lw=2)

end
plot!()