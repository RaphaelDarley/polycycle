using Flux
using Plots
using Statistics
using DataFrames
using CSV
using MLDatasets: BostonHousing

begin
    x = hcat(collect(Float32, -3:0.1:3)...)

    f(x) = @. 3x + 2
    y = f(x)
    x = x .* reshape(rand(Float32, 61), (1, 61))

    plot(vec(x), vec(y), lw=3, seriestype=:scatter, label="", title="Generated data", xlabel="x", ylabel="y")
end

begin
    df = DataFrame(CSV.File("A:/_Coding/Julia/_data/Life_Expectancy_Data.csv"))
    colnames = Symbol[]
    for i in string.(names(df))
        push!(colnames, Symbol(replace(replace(replace(strip(i), " " => "_"), "-" => "_"), "/" => "_")))
    end
    rename!(df, colnames)


    x = df.Adult_Mortality
    y = df.Life_expectancy
    # y_norm = y / maximum(y)
end

begin
    dataset = BostonHousing()
    x, y = BostonHousing(as_df=false)[:]
    x_train, x_test, y_train, y_test = x[:, 1:400], x[:, 401:end], y[:, 1:400], y[:, 401:end]
    x_train_n = Flux.normalise(x_train)
end;


custom_model(W, b, x) = @. W * x + b

W = rand(Float32, 1, 1)
W = [0.0f0]
b = [0.0f0]





function custom_loss(W, b, x, y)
    ŷ = custom_model(W, b, x)
    sum((y .- ŷ) .^ 2) / length(x)
end;
custom_loss(W, b, x, y)

function train_custom_model()
    dLdW, dLdb, _, _ = gradient(custom_loss, W, b, x, y)
    @. W = W - 0.1 * dLdW
    @. b = b - 0.1 * dLdb
    # plot!((x) -> b[1] + W[1] * x, 0, 800, label="Custom model", lw=2)
end;
train_custom_model();

for i = 1:40
    train_custom_model()
end


begin
    # plot(reshape(x, (61, 1)), reshape(y, (61, 1)), lw=3, seriestype=:scatter, label="", title="Simple Linear Regression", xlabel="x", ylabel="y")
    scatter(x, y)
    plot!((x) -> b[1] + W[1] * x, minimum(x), maximum(x), label="Custom model", lw=2)

end