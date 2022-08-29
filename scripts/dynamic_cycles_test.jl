using Flux
using Flux: update!
using Plots
using Statistics
using DataFrames
using CSV
using Optimisers
using XLSX

begin # import from CSV
    df = DataFrame(CSV.File("A:/_Coding/Julia/polycycle/oil_prod.csv"))
    y = df.oil_prod
    y_n = y ./ maximum(y)
    x = 1:size(y)[1]
end

hubbert(θ, x) = @. 2θ[1] / (1 + cosh(θ[2] * (x - θ[3])))



begin #hubbert model
    predict(θ, x) = @. 2θ[1] / (1 + cosh(θ[2] * (x - θ[3])))
    θ = [[1.0f0], [0.1f0], [maximum(x) * rand()]]
end


begin

    predict(θ, x) = hubbert(θ[1], x) .+ hubbert(θ[2], x)

    # predict(θ, x) = reduce(.+, hubbert(p, x) for p in θ)

    θ = [[[1.0f0], [0.1f0], [maximum(x) * rand()]], [[1.0f0], [0.1f0], [maximum(x) * rand()]]]
end


macro n_hubbert_cycles(n)
    return Meta.parse(join([:(hubbert(θ[$i], x)) for i in 1:n], " .+ "))
end

macro n_hubbert_cycles(n)
    cycle_arr = []
    for i in 1:n
        push!(cycle_arr, :(hubbert(θ[$i], x)))
    end
    return Meta.parse(join(cycle_arr, " .+ "))
end

function n_hubbert_θ(n; p1=1.0f0, p2=0.2f0, p3=(i) -> maximum(x) * rand())
    return [[p1, p2, p3(i)] for i in 1:n]
end

@macroexpand @n_hubbert_cycles(2)



predict(θ, x) = reduce(.+, hubbert(p, x) for p in θ)


# begin
n = 2
predict(θ) = @n_hubbert_cycles(2)
@macroexpand predict(θ) = @n_hubbert_cycles(2)
θ = n_hubbert_θ(n)
# end



@time (h, θ, loss, loss_plot, fit_plot) = fit_h(predict, θ, x, y_n;)

add_cyle_decomp!(fit_plot, θ)

display(fit_plot)

function add_cyle_decomp!(plot, θ)
    for (i, θi) in enumerate(θ)
        plot!((x) -> (hubbert(θi, x)[1]), minimum(x), maximum(x) * 2, label="decomp $i", lw=2)
    end
end


function fit_h(h::Function, θ, x, y; op_fun=Optimisers.AdaMax, kepochs=50, loss=Flux.Losses.mse)
    opt = Optimisers.setup(op_fun(), θ)
    J(θ, x, y) = loss(h(θ, x), y)
    cost_arr = []

    for i = 1:kepochs
        run_iters(1000, opt, J, θ, x, y_n)
        push!(cost_arr, J(θ, x, y))
    end
    loss_plot = plot(1:size(cost_arr)[1], cost_arr, title="Loss over iterations", lw=2)
    fit_plot = plot(x, y ./ maximum(y), size=(1200, 800), label="US annual Gas prod")
    plot!(fit_plot, (x) -> (h(θ, x)[1]), minimum(x), maximum(x) * 2, label="Custom model", lw=2)

    return (h, θ, loss, loss_plot, fit_plot)
end


function train_iter(opt, J, θ, x, y)
    loss, dJdθ = Flux.withgradient(θ -> J(θ, x, y), θ)
    opt, θ = Optimisers.update!(opt, θ, dJdθ[1])
end

function run_iters(n, opt, J, θ, x, y)
    for i = 1:n
        train_iter(opt, J, θ, x, y)
    end
end;


function fix_hubbert_params!(θ) # only works for one curve
    # θ[1] shouldn't be much bigger than ~1.5
    # θ[1] shouldn't be less than 0
    if (θ[1] < 0.0001)
        θ[1] = 1.0f0
    end
    # θ[2] should be greater than 0.001
    if θ[2] < 0.001
        θ[2] = 0.2f0
    end
end

# function fix_hubbert_params!(θ) # only works for one curve
#     # θ[1] shouldn't be much bigger than ~1.5
#     # θ[1] shouldn't be less than 0
#     v = (θ[1] > 0.0001)
#     θ[1] = v * θ[2] + !v * 1.0f0
#     # θ[2] should be greater than 0.001
#     v = θ[2] > 0.001
#     θ[2] = v * θ[2] + !v * 0.2f0
# end