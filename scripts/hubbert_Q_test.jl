using Plots

hubbert(θ, x) = @. 2θ[1] / (1 + cosh(θ[2] * (x - θ[3])))

hubbert(θ, x) = @. 2θ[1] / (1 + cosh(θ[2] * (x - θ[3])))

hubbertQ(θ, x) = @. 4 * θ[1] / (θ[2] * (1 + exp(θ[2] * (θ[3] - x))))

θ = [1.0f0, 0.1f0, 100]

X = 1:200
P = [hubbert(θ, x) for x in X]
Q = [P[1]]
for i in X
    if i == 1
        continue
    end
    # println(i)
    push!(Q, Q[i-1] + P[i])
end



plot(X, P, size=(1200, 800))
plot!(X, Q)

plot!((x) -> hubbertQ(θ, x), 0, 200)
plot!((x) -> hubbert(θ, x), 0, 200)