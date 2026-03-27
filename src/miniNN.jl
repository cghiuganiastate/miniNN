module miniNN

using Zygote: Params, withgradient

# --- Simple deterministic random number generator (LCG) ---
mutable struct SimpleRNG
    state::UInt32
end

SimpleRNG(seed::Integer=12345) = SimpleRNG(UInt32(seed))

function next_int(rng::SimpleRNG)
    rng.state = rng.state * 0x41c64e6d + 0x3039
    return rng.state
end

function next_float(rng::SimpleRNG)
    return next_int(rng) / UInt32(0xffffffff)
end

# --- Deterministic normal distribution approximation (Box-Muller transform) ---
function next_gaussian(rng::SimpleRNG)
    u1 = next_float(rng)
    u2 = next_float(rng)
    while u1 == 0.0 || u2 == 0.0
        u1 = next_float(rng)
        u2 = next_float(rng)
    end
    z0 = sqrt(-2.0 * log(u1)) * cos(2.0π * u2)
    return z0
end

# --- Generate random array using simple RNG ---
function randn_simple(::Type{T}, dims...; seed::Integer=12345) where T<:AbstractFloat
    rng = SimpleRNG(seed)
    return T.(next_gaussian(rng) for _ in 1:prod(dims)) |> x -> reshape(x, dims)
end

# --- Fisher-Yates shuffle using simple RNG ---
function shuffle_simple(rng::SimpleRNG, a::AbstractVector)
    b = collect(a)  # Convert to mutable vector
    for i in length(b):-1:2
        j = Int(floor(next_float(rng) * i)) + 1
        if j > i
            j = i
        end
        b[i], b[j] = b[j], b[i]
    end
    return b
end

# --- Activation functions ---
relu(x) = max.(0, x)
tanh_act(x) = tanh.(x)
sigmoid(x) = 1.0f0 ./ (1.0f0 .+ exp.(-x))
leakyrelu(x; α=0.01f0) = ifelse.(x .> 0, x, α .* x)
gelu(x) = 0.5f0 .* x .* (1 .+ erf.(x ./ sqrt(2f0)))
quickgelu(x) = x .* (1.0f0 ./ (1.0f0 .+ exp.(-1.702f0 .* x)))

# --- Kaiming Initialization ---
function kaiming_init(::Type{T}, fan_in::Int, fan_out::Int; seed::Integer=12345) where T<:AbstractFloat
    std = sqrt(T(2) / fan_in)
    rng = SimpleRNG(seed)
    return T.(std * next_gaussian(rng) for _ in 1:(fan_out * fan_in)) |> x -> reshape(x, fan_out, fan_in)
end

# --- Activation dictionary ---
const ACTIVATIONS = Dict(
    :relu      => relu,
    :tanh      => tanh_act,
    :sigmoid   => sigmoid,
    :leakyrelu => x -> leakyrelu(x; α=0.01f0),
    :gelu      => gelu,
    :quickgelu => quickgelu,
    :identity  => identity
)

# --- Dense Layer ---
struct Dense
    W::Matrix{Float32}
    b::Vector{Float32}
    σ::Function
end

Dense(in::Int, out::Int, σ=identity) = Dense(
    kaiming_init(Float32, in, out),
    zeros(Float32, out),
    σ
)

(layer::Dense)(x::AbstractVector) = layer.σ.(layer.W * x .+ layer.b)
(layer::Dense)(x::AbstractMatrix) = layer.σ.(layer.W * x .+ layer.b)

# --- Scaler Layer ---
struct Scaler
    μ::Vector{Float32}
    σ::Vector{Float32}
end

function Scaler(X::AbstractMatrix)
    μ = Float32.(rowmean(X))
    σ = Float32.(rowstd(X))
    σ[σ .< 1f-3] .= 1f0
    return Scaler(μ, σ)
end

(s::Scaler)(x::AbstractVector) = (x .- s.μ) ./ s.σ
(s::Scaler)(x::AbstractMatrix) = (x .- s.μ) ./ s.σ

# --- Inverse Scaler ---
struct InverseScaler
    μ::Vector{Float32}
    σ::Vector{Float32}
end

(inv::InverseScaler)(x::AbstractVector) = x .* inv.σ .+ inv.μ
(inv::InverseScaler)(x::AbstractMatrix) = x .* inv.σ .+ inv.μ

# --- Chain ---
struct Chain
    layers::Vector{Any}
end

(c::Chain)(x) = foldl((a, f) -> f(a), c.layers; init=x)

# --- Collect Parameters ---
params(model) = vcat(
    [layer.W for layer in model.layers if layer isa Dense],
    [layer.b for layer in model.layers if layer isa Dense],
)

# --- Mean & Loss ---
mean(v) = sum(v) / length(v)
mse(ŷ, y) = mean((ŷ .- y).^2)

# ==========================================================
# --- Optimizers ---
mutable struct Adam
    η::Float32; β1::Float32; β2::Float32; ϵ::Float32
    m::IdDict; v::IdDict; t::Int
end

function Adam(η=0.001; β1=0.9, β2=0.999, ϵ=1e-8)
    Adam(Float32(η), Float32(β1), Float32(β2), Float32(ϵ),
         IdDict(), IdDict(), 0)
end

function update!(opt::Adam, ps, grads)
    opt.t += 1
    for p in ps
        g = grads[p]; if g === nothing; continue; end
        m = get!(opt.m, p, zero(p))
        v = get!(opt.v, p, zero(p))
        m .= opt.β1 .* m .+ (1 - opt.β1) .* g
        v .= opt.β2 .* v .+ (1 - opt.β2) .* (g .^ 2)
        m̂ = m ./ (1 - opt.β1^opt.t)
        v̂ = v ./ (1 - opt.β2^opt.t)
        p .-= opt.η .* m̂ ./ (sqrt.(v̂) .+ opt.ϵ)
    end
end

mutable struct SGD
    η::Float32
    SGD(η=0.01) = new(Float32(η))
end



function update!(opt::SGD, ps, grads)
    for p in ps
        g = grads[p]; if g === nothing; continue; end
        p .-= opt.η .* g
    end
end
# ==========================================================

# --- Make minibatches with shuffling ---
function make_batches(X, Y, batchsize=128; shuffle=true, rng_seed::Integer=12345)
    n = size(X, 2)
    idxs = shuffle ? shuffle_simple(SimpleRNG(rng_seed), 1:n) : 1:n
    parts = Iterators.partition(idxs, batchsize)
    [(X[:,i], Y[:,i]) for i in parts]
end

# --- Training Step (minibatch aware) ---
function train!(model, loss, data, opt=Adam(); verbose=true, epoch=0)
    total_loss = 0.0f0
    
    # Check if model has InverseScaler (last layer)
    has_inv_scaler = model.layers[end] isa InverseScaler
    inv_scaler = has_inv_scaler ? model.layers[end] : nothing
    
    for (Xbatch, Ybatch) in data
        ps = Params(params(model))
        l, grads = withgradient(ps) do
            ŷ = model(Xbatch)
            # If model has InverseScaler, we need to scale output back down for loss
            ŷ_scaled = has_inv_scaler ? (ŷ .- inv_scaler.μ) ./ inv_scaler.σ : ŷ
            loss(ŷ_scaled, Ybatch)
        end
        total_loss += l
        update!(opt, ps, grads)
    end
    if verbose
        println("Epoch $(epoch): Loss = $(round(total_loss / length(data), digits=6))")
    end
end

# --- Rowwise utilities ---
function rowmean(X::AbstractMatrix)
    n = size(X, 2)
    sum(X, dims=2)[:] / n
end

function rowstd(X::AbstractMatrix)
    μ = rowmean(X)
    n = size(X, 2)
    sqrt.(sum((X .- μ).^2, dims=2)[:] / n)
end

# --- Convenience Constructor ---
function build_model(X, Y; hidden, verbose=true,
                     scaling=:inputs, lr_epochs=[100], lr_values=[0.001],
                     optimizer=:adam, activations=nothing,
                     batchsize=128, shuffle=true)

    layers = Any[]
    local xscaler, yscaler, yinv

    if scaling == :inputs || scaling == :both
        xscaler = Scaler(X)
        push!(layers, xscaler)
    end

    if scaling == :both
        yscaler = Scaler(Y)
        yinv    = InverseScaler(yscaler.μ, yscaler.σ)
    else
        # When not scaling outputs, use identity transform (μ=0, σ=1)
        yscaler = Scaler(zeros(Float32, size(Y,1), size(Y,2)))  # μ=0
        yinv    = InverseScaler(zeros(Float32, size(Y,1)), ones(Float32, size(Y,1)))  # μ=0, σ=1
    end

    if activations === nothing
        activations = [ (i < length(hidden)-1 ? :relu : :identity)
                        for i in 1:(length(hidden)-1) ]
    end

    for (i, (in_dim, out_dim)) in enumerate(zip(hidden[1:end-1], hidden[2:end]))
        σ = get(ACTIVATIONS, activations[i], identity)
        push!(layers, Dense(in_dim, out_dim, σ))
    end
    
    # Only add inverse scaler if we're scaling outputs
    if scaling == :both
        push!(layers, yinv)
    end

    model = Chain(layers)

    total_epochs = lr_epochs[end]

    # Initialize optimizer with first learning rate value
    opt = optimizer == :adam ? Adam(lr_values[1]) :
          optimizer == :sgd  ? SGD(lr_values[1])  :
          error("Unknown optimizer: $optimizer")

    for epoch in 1:total_epochs
        # Get learning rate for current epoch based on schedule
        η_epoch = lr_values[1]
        for (cut, lr) in zip(lr_epochs, lr_values)
            if epoch <= cut
                η_epoch = lr
                break
            end
        end
        if optimizer == :adam
            opt.η = Float32(η_epoch)
        elseif optimizer == :sgd
            opt.η = Float32(η_epoch)
        end

        # Create shuffled minibatches each epoch
        # Note: when scaling=:both, targets are scaled, model outputs unscaled via InverseScaler
        # So we need to scale targets AND also scale model outputs for loss computation
        data = if scaling == :both
            # Scale targets for training
            Y_scaled = (Y .- yscaler.μ) ./ yscaler.σ
            make_batches(X, Y_scaled, batchsize; shuffle=shuffle, rng_seed=12345+epoch)
        else
            make_batches(X, Y, batchsize; shuffle=shuffle, rng_seed=12345+epoch)
        end

        train!(model, mse, data, opt; verbose=verbose, epoch=epoch)
    end

    return model
end

end # module
