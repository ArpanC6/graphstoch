module GraphStoch
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