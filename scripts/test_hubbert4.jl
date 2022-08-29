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

begin # import gas data from sheet
    xf = XLSX.readxlsx("polycycle_data.xlsx")
    oil_sheet = xf["US gas prod"]
    df = XLSX.eachtablerow(oil_sheet) |> DataFrames.DataFrame

    y = df.gas_prod
    y_n = y ./ maximum(y)
    x = 1:size(y)[1]
end


begin #hubbert model
    θ = [[1.0f0], [0.1f0], [maximum(x) / 2]]
    predict(θ, x) = @. 2θ[1] / (1 + cosh(θ[2] * (x - θ[3])))
end

# begin #hubbert model
#     hubbert(θ, x) = @. 2θ[1] / (1 + cosh(θ[2] * (x - θ[3])))
#     gen_hubbert_θ() = [[1.0f0], [0.1f0], [maximum(x) * rand()]]

#     predict(θ, x) = @. hubbert(θ[1], x) + hubbert(θ[2], x)

#     θ = [gen_hubbert_θ(), gen_hubbert_θ()]
# end

begin # double hubbert
    predict(θ, x) = @. 2θ[1] / (1 + cosh(θ[2] * (x - θ[3]))) + 2θ[4] / (1 + cosh(θ[5] * (x - θ[6])))

    θ = [[1.0f0], [0.1f0], [maximum(x) * 0.3], [1.0f0], [0.1f0], [maximum(x)] * 0.6]
    θ_gen() = [[1.0f0], [0.1f0], [maximum(x) * rand()], [1.0f0], [0.1f0], [maximum(x)] * rand()]

end

begin # tripple hubbert
    predict(θ, x) = @. 2θ[1] / (1 + cosh(θ[2] * (x - θ[3]))) + 2θ[4] / (1 + cosh(θ[5] * (x - θ[6]))) + 2θ[7] / (1 + cosh(θ[8] * (x - θ[9])))
    θ_gen() = [[1.0f0], [0.1f0], [maximum(x) * rand()], [1.0f0], [0.1f0], [maximum(x)] * rand(), [1.0f0], [0.1f0], [maximum(x)] * rand()]
    θ = [[1.0f0], [0.75f0], [70], [1.0f0], [0.75f0], [85], [1.0f0], [0.75f0], [120]]
end

# begin
#     hubbert(θ, x) = @. 2θ[1] / (1 + cosh(θ[2] * (x - θ[3])))

#     predict(θ, x) = @. hubbert(θ[1:3], x) + hubbert(θ[4:6], x)

#     θ = [[1.0f0], [0.1f0], [maximum(x) * rand()], [1.0f0], [0.1f0], [maximum(x)] * rand()]
# end

(h, θ, loss, loss_plot, fit_plot) = fit_h(predict, θ, x, y_n;)
println(fit_h(predict, θ, x, y_n))

for j = 1:Int(size(θ)[1] / 3)
    println(j)
    start_i = (j - 1) * 3 + 1
    plot!((x) -> (hubbert(θ[start_i:start_i+2], x)[1]), minimum(x), maximum(x) * 2, label="decomp $j", lw=2)
end

display(loss_plot)
display(fit_plot)
png(fit_plot, "US NG 3 cycle - test")

@time fit_n_times(1, predict, θ_gen, x, y_n; op_fun=Optimisers.OAdam)
@time fit_n_times(1, predict, θ_gen, x, y_n; op_fun=Optimisers.AdaMax)

function fit_n_times(n, h::Function, θ_gen, x, y; op_fun=Optimisers.AdaMax, kepochs=50, loss=Flux.Losses.mse)
    acc = []
    for i in 1:n
        push!(acc, fit_h(h, θ_gen(), x, y; op_fun, kepochs, loss))
    end
    fit_plot = scatter(x, y, legend=:topleft, size=(1200, 1200))
    for (i, res) = enumerate(acc)
        # display(res[5])
        plot!(fit_plot, (x) -> (res[1](res[2], x)[1]), minimum(x), maximum(x) * 2, label="run $i", lw=2)
        for j = 1:Int(size(res[2])[1] / 3)
            println(j)
            start_i = (j - 1) * 3 + 1
            plot!(fit_plot, (x) -> (hubbert(res[2][start_i:start_i+2], x)[1]), minimum(x), maximum(x) * 2, label="run $i", lw=2)
        end
    end
    return fit_plot
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