using Flux
using DataFrames
using CSV
using Plots
using Lathe
using GLM
using Statistics
using StatsPlots
using MLBase
using Flux: update!


a = rand()
b = rand()

predict(a, b) = (a .* df.Adult_Mortality) .+ b
predict(x) = (a * x) + b

loss(a, b) = sum((predict(a, b) .- df.Life_expectancy) .^ 2) / size(df)[1]

# x, y = rand(5), rand(2)



df = DataFrame(CSV.File("A:/_Coding/Julia/_data/Life_Expectancy_Data.csv"))
colnames = Symbol[]
for i in string.(names(df))
    push!(colnames, Symbol(replace(replace(replace(strip(i), " " => "_"), "-" => "_"), "/" => "_")))
end
rename!(df, colnames);

scatter(df.Adult_Mortality, df.Life_expectancy,
    title="Scatter Plot Life Expectancy vs Adult Mortality Rate",
    ylabel="Life Expectancy",
    xlabel="Adult Mortality Rate",
    legend=false)

plot!(predict, 0, 800)





η = 0.00001

a = -0.05
b = 75

a, b = 0, 0


grads = gradient(loss, a, b)
a -= η * grads[1]
b -= η * grads[2]
plot!(predict, 0, 800)
