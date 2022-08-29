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

begin
    hubbert(θ, x) = @. 2θ[1] / (1 + cosh(θ[2] * (x - θ[3])))

    predict(θ, x) = reduce(.+, hubbert(p, x) for p in θ)

    θ_gen(n=1; p1=(i) -> 1.0f0, p2=(i) -> 0.2f0, p3=(i) -> maximum(x) * rand()) = [[p1(i), p2(i), p3(i)] for i in 1:n]
end

@time (h, θ, loss) = fit_h(predict, θ_gen(5), x, y_n;)
fit_plot = plot_fit(x, y_n, predict, θ)
add_cyle_decomp!(fit_plot, θ)
display(fit_plot)


@time res = fit_n_times(1, predict, θ_gen, x, y_n; op_fun=Optimisers.OAdam)
@time fit_n_times(1, predict, θ_gen, x, y_n; op_fun=Optimisers.AdaMax)


# for i in 1:2
i = 2
θ_geni() = θ_gen(i)
fits = fit_n_times(5, predict, θ_geni, x, y_n)
best_fit = []
best_loss = Inf
for fit in fits
    new_loss = Flux.Losses.mse(predict(fit[2], x), y_n)
    if new_loss < best_loss
        best_fit = fit[2]
        best_loss = new_loss
    end
end
fit_plot = plot_fit(x, y_n, predict, best_fit)
display(fit_plot)
# end


begin
    fits = fit_n_times(1, predict, () -> θ_gen(), x, y_n)
    best_fit = []
    best_loss = Inf
    for fit in fits
        new_loss = Flux.Losses.mse(predict(fit[2], x), y_n)
        if new_loss < best_loss
            best_fit = fit[2]
            best_loss = new_loss
        end
    end
    fit_plot = plot_fit(x, y_n, predict, best_fit)
    display(fit_plot)
end


function fit_n_times(n, h::Function, θ_gen, x, y; op_fun=Optimisers.AdaMax, kepochs=50, loss=Flux.Losses.mse)
    acc = []
    for i in 1:n
        push!(acc, fit_h(h, θ_gen(), x, y; op_fun, kepochs, loss))
    end
    return acc
end

function fit_h(h::Function, θ, x, y; op_fun=Optimisers.AdaMax, kepochs=50, loss=Flux.Losses.mse)
    opt = Optimisers.setup(op_fun(), θ)
    J(θ, x, y) = loss(h(θ, x), y)

    ke_count = 0
    reset_count = 0
    while ke_count < kepochs
        run_iters(1000, opt, J, θ, x, y_n)
        if valid_θ(θ)
            ke_count += 1
        else
            θ = θ_gen(size(θ)[1])
            ke_count = 0
            reset_count += 1
            println("invalid parameters: resetting ($reset_count)")
            # if reset_count > 10
            #     break
            # end
        end
    end
    final_loss = J(θ, x, y)
    return (h, θ, final_loss)
end




function train_iter(opt, J, θ, x, y)
    _, dJdθ = Flux.withgradient(θ -> J(θ, x, y), θ)
    opt, θ = Optimisers.update!(opt, θ, dJdθ[1])
end





## utility

function run_iters(n, opt, J, θ, x, y)
    for i = 1:n
        train_iter(opt, J, θ, x, y)
    end
end;

function valid_θ(θ)
    return all([valid_hubbert_params(θi) for θi in θ])
end

function valid_hubbert_params(θ)
    return (
        # θ[1] shouldn't be much bigger than ~1.5; peak to high
        θ[1] < 1.4
        # θ[1] shouldn't be less than 0; would be negative curve
        && θ[1] > 0.0001
        # θ[2] should be greater than 0.001; to flat
        && θ[2] > 0.001
        && !(NaN in θ)
    )
end

## graphing functions

function add_cyle_decomp!(fit_plot, θ)
    for (i, θi) in enumerate(θ)
        plot!(fit_plot, (x) -> (hubbert(θi, x)[1]), minimum(x), maximum(x) * 2, label="decomp $i", lw=2)
    end
end

function plot_fit(x, y, h, θ)
    fit_plot = plot(x, y, size=(1200, 800), label="true data", lw=2)
    plot!(fit_plot, (x) -> (h(θ, x)[1]), minimum(x), maximum(x) * 2, label="Custom model", lw=2)
    return fit_plot
end
