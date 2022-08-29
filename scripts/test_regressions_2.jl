using Flux
using CSV
using DataFrames
using Plots

#https://gist.githubusercontent.com/aishwarya8615/89d9f36fc014dea62487f7347864d16a/raw/8629d284e13976dcb13bb0b27043224b9266fffa/Life_Expectancy_Data.csv
df = DataFrame(CSV.File("A:/_Coding/Julia/_data/Life_Expectancy_Data.csv"))
colnames = Symbol[]
for i in string.(names(df))
    push!(colnames, Symbol(replace(replace(replace(strip(i), " " => "_"), "-" => "_"), "/" => "_")))
end
rename!(df, colnames);


X = df.Adult_Mortality
Y = df.Life_expectancy

scatter(X, Y)

θ = [-0.05, 65]
params = Flux.params(θ)

h(x) = x .* θ[1] .+ θ[2]
J(x, y) = sum((h(x) .- y) .^ 2) / size(x)[1]

J(X, Y)

gradient(() -> J(X, Y))