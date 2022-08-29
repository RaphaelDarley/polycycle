using Flux

params = [1, 2]

f(params, x) = x * params[1] + x * params[2]

gradient(f, params, 1)