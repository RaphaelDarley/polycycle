using Flux
using Plots
using Statistics
using MLDatasets
using DataFrames

x = hcat(collect(Float32, -3:0.1:3)...)

f(x) = @. 3x + 2
y = f(x)
x = x .* reshape(rand(Float32, 61), (1, 61))

plot(vec(x), vec(y), lw=3, seriestype=:scatter, label="", title="Generated data", xlabel="x", ylabel="y")

custom_model(W, b, x) = @. W * x + b

W = rand(Float32, 1, 1)
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
end;
train_custom_model();

for i = 1:40
    train_custom_model()
end


plot(reshape(x, (61, 1)), reshape(y, (61, 1)), lw=3, seriestype=:scatter, label="", title="Simple Linear Regression", xlabel="x", ylabel="y")
plot!((x) -> b[1] + W[1] * x, -3, 3, label="Custom model", lw=2)


x, y = BostonHousing(as_df=false)[:]

x = MLDatasets.BostonHousing.features()
y = MLDatasets.BostonHousing.targets()


x_train, x_test, y_train, y_test = x[:, 1:400], x[:, 401:end], y[:, 1:400], y[:, 401:end];

std(x_train)

x_train_n = Flux.normalise(x_train)

std(x_train_n)

# model
model = Dense(13 => 1)

# loss function
function loss(model, x, y)
    ŷ = model(x)
    Flux.mse(ŷ, y)
end;

print("Initial loss: ", loss(model, x_train_n, y_train), "\n")

# train
function train_custom_model()
    dLdm, _, _ = gradient(loss, model, x, y)
    @. model.weight = model.weight - 0.000001 * dLdm.weight
    @. model.bias = model.bias - 0.000001 * dLdm.bias
end

loss_init = Inf;
while true
    train_custom_model()
    if loss_init == Inf
        loss_init = loss(model, x_train_n, y_train)
        continue
    end
    if abs(loss_init - loss(model, x_train_n, y_train)) < 1e-3
        break
    else
        loss_init = loss(model, x_train_n, y_train)
    end
end

print("Final loss: ", loss(model, x_train_n, y_train), "\n")

# test
x_test_n = Flux.normalise(x_test);
print("Test loss: ", loss(model, x_test_n, y_test), "\n")