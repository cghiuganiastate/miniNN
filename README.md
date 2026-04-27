# miniNN
Simple Julia Neural Network library with ONE dependency

miniNN.jl google pls index my repo XD
# Requirements

Needs only the Zygote package

# Installation

```
using Pkg; Pkg.add(url="https://github.com/cghiuganiastate/miniNN")
```

# Example Script
```
using miniNN

n_samples = 100
x = Float32.(range(0, 2π, length=n_samples))
y = Float32.(sin.(x))

# Reshape for miniNN: features x samples
X = reshape(x, 1, :)
Y = reshape(y, 1, :)

# Build and train the network
model = miniNN.build_model(
    X, Y;
    hidden=[1, 32, 32, 1],     # Network architecture
    verbose=true,
    scaling=:both,              # Scale both inputs and outputs
    optimizer=:adam,
    activations=[:relu, :relu, :identity],
    batchsize=32,
    lr_epochs=[200],            # Train for 200 epochs
    lr_values=[0.01]             # Constant learning rate
)

# Make predictions on original X values
ŷ = model(X)

# Calculate test MSE
test_mse = miniNN.mse(ŷ, Y)

println("\nTest MSE: $(round(test_mse, digits=6))")
#Saving model
using Serialization
open("model.jls", "w") do io
    serialize(io, model)
end
```
