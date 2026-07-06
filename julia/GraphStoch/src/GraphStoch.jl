module GraphStoch
using StochasticDiffEq
using LinearAlgebra

# Build a random graph with n nodes using the Erdos-Renyi model
function random_graph(n::Int, p::Float64)
    A = zeros(n, n)
    for i in 1:n, j in (i + 1):n
        if rand() < p
            A[i, j] = 1.0
            A[j, i] = 1.0
        end
    end
    return A
end

# Build the degree matrix D from an adjacency matrix A.
# D is diagonal, where D[i,i] equals the degree of node i.
function degree_matrix(A::AbstractMatrix)
    n = size(A, 1)
    D = zeros(n, n)
    for i in 1:n
        D[i, i] = sum(A[i, :])
    end
    return D
end

# Graph Laplacian: L = D - A
# Captures the connectivity structure of the graph.
function graph_laplacian(A::AbstractMatrix)
    D = degree_matrix(A)
    return D - A
end

# Simulate deterministic graph diffusion:
# dX/dt = -L * X
function diffuse(L::AbstractMatrix, X0::AbstractVector, dt::Float64, steps::Int)
    X = copy(X0)
    history = [copy(X)]
    for _ in 1:steps
        X = X .+ dt .* (-L * X)
        push!(history, copy(X))
    end
    return history
end

# Euler-Maruyama solver for the stochastic diffusion equation:
# dX_t = -L * X_t * dt + sigma * dW_t
function sde_solve(L::AbstractMatrix, X0::AbstractVector, dt::Float64, steps::Int, sigma::Float64)
    n = length(X0)
    X = copy(X0)
    history = [copy(X)]
    for _ in 1:steps
        Z = randn(n)
        X = X .+ dt .* (-L * X) .+ sigma * sqrt(dt) .* Z
        push!(history, copy(X))
    end
    return history
end

# One-step neighbor averaging (single GCN-like layer)
# X_new[i] = average of node i and all of its neighbors
function naive_gnn_denoise(A::AbstractMatrix, X_noisy::AbstractVector)
    n = length(X_noisy)
    X_new = copy(X_noisy)
    for i in 1:n
        neighbors = findall(x -> x == 1.0, A[i, :])
        if !isempty(neighbors)
            X_new[i] = (X_noisy[i] + sum(X_noisy[neighbors])) /
                       (1 + length(neighbors))
        end
    end
    return X_new
end

# Repeat neighbor averaging for multiple iterations so that
# the comparison with GraphStoch is fair.
function naive_gnn_multi(A::AbstractMatrix, X_noisy::AbstractVector, iterations::Int)
    X = copy(X_noisy)
    for _ in 1:iterations
        X = naive_gnn_denoise(A, X)
    end
    return X
end

# Mean Squared Error (MSE)
function mse(x_true::AbstractVector, x_pred::AbstractVector)
    return sum((x_true .- x_pred).^2) / length(x_true)
end

# Solve the same SDE using StochasticDiffEq's SRA3 (adaptive, additive-noise
# specific solver), for comparison against the from-scratch Euler-Maruyama.
function sra3_solve(L::AbstractMatrix, X0::AbstractVector, sigma::Float64, tspan::Tuple{Float64,Float64})
    f(X, p, t) = -L * X
    g(X, p, t) = fill(sigma, length(X))
    prob = SDEProblem(f, g, X0, tspan)
    sol = solve(prob, SRA3())
    return sol
end

# Matrix-input overloads (for multi-feature node states, e.g. Cora/Citeseer/PubMed)
# Each column of X0/X_noisy is treated as one independent feature diffused
# over the same graph structure. Multiple dispatch picks these automatically
# when a matrix (not a vector) is passed in.

function diffuse(L::AbstractMatrix, X0::AbstractMatrix, dt::Float64, steps::Int)
    X = copy(X0)
    history = [copy(X)]
    for _ in 1:steps
        X = X .+ dt .* (-L * X)
        push!(history, copy(X))
    end
    return history
end

function sde_solve(L::AbstractMatrix, X0::AbstractMatrix, dt::Float64, steps::Int, sigma::Float64)
    X = copy(X0)
    history = [copy(X)]
    for _ in 1:steps
        Z = randn(size(X))
        X = X .+ dt .* (-L * X) .+ sigma * sqrt(dt) .* Z
        push!(history, copy(X))
    end
    return history
end

function naive_gnn_denoise(A::AbstractMatrix, X_noisy::AbstractMatrix)
    n = size(X_noisy, 1)
    X_new = copy(X_noisy)
    for i in 1:n
        neighbors = findall(x -> x == 1.0, A[i, :])
        if !isempty(neighbors)
            X_new[i, :] = (X_noisy[i, :] .+ vec(sum(X_noisy[neighbors, :], dims=1))) ./
                          (1 + length(neighbors))
        end
    end
    return X_new
end

function naive_gnn_multi(A::AbstractMatrix, X_noisy::AbstractMatrix, iterations::Int)
    X = copy(X_noisy)
    for _ in 1:iterations
        X = naive_gnn_denoise(A, X)
    end
    return X
end

function mse(x_true::AbstractMatrix, x_pred::AbstractMatrix)
    return sum((x_true .- x_pred).^2) / length(x_true)
end

function exact_solve(L::AbstractMatrix, X0::AbstractVector, sigma::Float64, t::Float64)
    n = length(X0)
    F = eigen(Symmetric(L))
    λ = F.values
    U = F.vectors

    mean_t = U * (exp.(-λ .* t) .* (U' * X0))

    var_modes = [λ_i ≈ 0.0 ? sigma^2 * t :
                 sigma^2 * (1 - exp(-2 * λ_i * t)) / (2 * λ_i) for λ_i in λ]

    Z = randn(n)
    sample_t = mean_t .+ U * (sqrt.(var_modes) .* Z)

    return sample_t, mean_t, var_modes
end

function exact_solve(L::AbstractMatrix, X0::AbstractMatrix, sigma::Float64, t::Float64)
    n, k = size(X0)
    F = eigen(Symmetric(L))
    λ = F.values
    U = F.vectors

    mean_t = U * (exp.(-λ .* t) .* (U' * X0))

    var_modes = [λ_i ≈ 0.0 ? sigma^2 * t :
                 sigma^2 * (1 - exp(-2 * λ_i * t)) / (2 * λ_i) for λ_i in λ]

    Z = randn(n, k)
    sample_t = mean_t .+ U * (sqrt.(var_modes) .* Z)

    return sample_t, mean_t, var_modes
end
"""
    diffuse_with_teleport(X0, L, alpha, sigma, dt, n_steps)

Euler-Maruyama simulation of the teleport-augmented OU process:
    dX_t = -(L + alpha*I) X_t dt + alpha*X0 dt + sigma dW_t
"""
function diffuse_with_teleport(X0, L, alpha, sigma, dt, n_steps)
    n = size(X0, 1)
    M = L + alpha * I(n)
    X = copy(X0)
    for _ in 1:n_steps
        dW = sqrt(dt) * randn(size(X))
        X = X .+ (-M * X .+ alpha .* X0) .* dt .+ sigma .* dW
    end
    return X
end

"""
    diffuse_with_reaction(X0, L, beta, sigma, dt, n_steps)

Euler-Maruyama simulation of a reaction-diffusion SDE with a bistable
(Allen-Cahn-type) reaction term:
    dX_t = -L X_t dt + beta*(X_t - X_t.^3) dt + sigma dW_t

The reaction term prevents collapse to a single graph-wide constant
(oversmoothing) by pushing values away from 0 toward +/-1, independent
of any reference signal X0 - unlike the teleport term, this does not
require access to (noisy or clean) X0 to function.
"""
function diffuse_with_reaction(X0, L, beta, sigma, dt, n_steps)
    X = copy(X0)
    for _ in 1:n_steps
        dW = sqrt(dt) * randn(size(X))
        X = X .+ (-L * X .+ beta .* (X .- X.^3)) .* dt .+ sigma .* dW
    end
    return X
end
end # module GraphStoch