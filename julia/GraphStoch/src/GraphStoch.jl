module GraphStoch
using StochasticDiffEq

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

end # module GraphStoch