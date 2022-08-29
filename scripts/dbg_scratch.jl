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
    θ = [[1.0f0], [0.1f0], [maximum(x) / 2]]
    predict(θ, x) = @. 2θ[1] / (1 + cosh(θ[2] * (x - θ[3])))
end


h = predict
θ = θ
x = x
y = y_n
op_fun = Optimisers.AdaMax
kepochs = 50
loss = Flux.Losses.mse
opt = Optimisers.setup(op_fun(), θ)
J(θ, x, y) = loss(h(θ, x), y)
cost_arr = []

function train_iter(opt, J, θ, x, y)
    loss, dJdθ = Flux.withgradient(θ -> J(θ, x, y), θ)
    opt, θ = Optimisers.update!(opt, θ, dJdθ[1])
end

function run_iters(n, opt, J, θ, x, y)
    for i = 1:n
        train_iter(opt, J, θ, x, y)
    end
end;

for i = 1:kepochs
    run_iters(1000, opt, J, θ, x, y)
end

loss_plot = plot((x) -> (h(θ, x)[1]), minimum(x), maximum(x), title="Loss over ", kepochs * 1000, " iterations", lw=2)


macro test(n)
    return [Meta.parse(join([:(println($i)) for i in 1:n], " ; "))]
end

function foo()
    @test(3)
end

foo()