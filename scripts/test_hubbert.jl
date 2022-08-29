using Flux
using Flux: update!
using Plots
using Statistics
using DataFrames
using CSV

begin
    x = hcat(collect(Float32, -3:0.1:3)...)

    f(x) = @. 3x + 2
    y = f(x)
    x = x .* reshape(rand(Float32, 61), (1, 61))

    plot(vec(x), vec(y), lw=3, seriestype=:scatter, label="", title="Generated data", xlabel="x", ylabel="y")
end

begin # manual data input SA
    raw = "2219	2618	2825	3081	3262	3851	4821	6070	7693	8618	7216	8762	9419	8554	9842	10270	10256	6961	4951	4534	3601	5208	4450	5656	5636	7106	8820	9092	8893	8983	8974	9087	9005	9267	8524	9121	8935	8207	9628	10306	10839	10671	10269	10665	9709	9865	11079	11622	11393	11519	11998	12406	11892	12261	11832	11039	10954"
    split_str = split(raw, "\t")

    y = [parse(Float64, s) for s in split_str]
    x = 1:size(y)[1]
end

begin # import from CSV
    df = DataFrame(CSV.File("A:/_Coding/Julia/polycycle/oil_prod.csv"))
    y = df.oil_prod
    x = 1:size(y)[1]
end

begin #linear model
    linear_model(θ, x) = @. θ[1] * x + θ[2]
    θ = [[0.0f0], [0.0f0]]
end

begin #hubbert model
    hubbert_model(θ, x) = @. 2θ[1] / (1 + cosh(θ[2] * (x - θ[3])))
    # θ = [[1.0f0], [1.0f0], [1.0f0]]
    θ = [[1.0f0], [0.5f0], [maximum(x) / 2]]
end

plot((x) -> 0.25 * (sech(x / 2))^2)
plot((x) -> hubbert_model(θ, x)[1])

function custom_loss(h, θ, x, y)
    ŷ = h(θ, x)
    sum((y .- ŷ) .^ 2) / length(x)
end;



custom_loss(linear_model, θ, x, y ./ maximum(y))
custom_loss(hubbert_model, θ, x, y ./ maximum(y))

function train_custom_model()
    # dLdθ, _, _ = gradient(custom_loss, θ, x, y ./ maximum(y))
    dLdθ, _, _ = gradient((θ, x, y) -> custom_loss(hubbert_model, θ, x, y), θ, x, y ./ maximum(y))

    for (θj, dLdθj) in Iterators.zip(θ, dLdθ)
        # println(θj, dLdθj)
        @. θj -= 0.01 * dLdθj[1]
    end
    return custom_loss(hubbert_model, θ, x, y ./ maximum(y))
end;

train_custom_model();

cost_arr = []

for i = 1:1000
    push!(cost_arr, train_custom_model())
end


plot(1:size(cost_arr)[1], cost_arr)


begin
    # plot(reshape(x, (61, 1)), reshape(y, (61, 1)), lw=3, seriestype=:scatter, label="", title="Simple Linear Regression", xlabel="x", ylabel="y")
    scatter(x, y ./ maximum(y))
    plot!((x) -> (hubbert_model(θ, x)[1]), minimum(x), maximum(x), label="Custom model", lw=2)
end

plot!((x) -> (hubbert_model([[0.75f0], [0.08f0], [75.0f0]], x)[1]), minimum(x), maximum(x), label="Custom model", lw=2)