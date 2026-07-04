module GraphStoch
# Build a random graph with n nodes and edge probability p (Erdos-Renyi style)
function random_graph(n::Int, p::Float64)
    A = zeros(n, n)
    for i in 1:n, j in (i+1):n
        if rand() < p
            A[i, j] = 1.0
            A[j, i] = 1.0
        end
    end
    return A
end

# Naive baseline: one-step neighbor averaging (like a single GCN layer)
# Fair comparison: run naive neighbor-averaging for the SAME number of iterations
# as GraphStoch, instead of just one step.
function naive_gnn_multi(A::Matrix, X_noisy::Vector, iterations::Int)
    X = copy(X_noisy)
    for _ in 1:iterations
        X = naive_gnn_denoise(A, X)
    end
    return X
end
# X_new[i] = average of X[i] and its neighbors' values
function naive_gnn_denoise(A::Matrix, X_noisy::Vector)
    n = length(X_noisy)
    X_new = copy(X_noisy)
    for i in 1:n
        neighbors = findall(x -> x == 1.0, A[i, :])
        if !isempty(neighbors)
            X_new[i] = (X_noisy[i] + sum(X_noisy[neighbors])) / (1 + length(neighbors))
        end
    end
    return X_new
end

# Mean squared error between two vectors
function mse(x_true::Vector, x_pred::Vector)
    return sum((x_true .- x_pred).^2) / length(x_true)
end
# Euler-Maruyama SDE solver:
# dX_t = -L*X_t*dt + sigma*dW_t
# Adds random Brownian noise on top of the deterministic diffusion.
function sde_solve(L::Matrix, X0::Vector, dt::Float64, steps::Int, sigma::Float64)
    n = length(X0)
    X = copy(X0)
    history = [copy(X)]
    for _ in 1:steps
        Z = randn(n)  # standard normal random noise
        X = X .+ dt .* (-L * X) .+ sigma * sqrt(dt) .* Z
        push!(history, copy(X))
    end
    return history
end
# Simulate simple diffusion: dX/dt = -L*X
# Starting state X0 spreads across the graph over time using small steps.
function diffuse(L::Matrix, X0::Vector, dt::Float64, steps::Int)
    X = copy(X0)
    history = [copy(X)]
    for _ in 1:steps
        X = X .+ dt .* (-L * X)
        push!(history, copy(X))
    end
    return history
end
# Build the degree matrix D from an adjacency matrix A.
# D is diagonal, where D[i,i] = number of connections node i has.
function degree_matrix(A::Matrix)
    n = size(A, 1)
    D = zeros(n, n)
    for i in 1:n
        D[i, i] = sum(A[i, :])
    end
    return D
end

# Graph Laplacian: L = D - A
# This operator captures how much each node differs from its neighbors.
function graph_laplacian(A::Matrix)
    D = degree_matrix(A)
    return D - A
end

end # module GraphStoch