import numpy as np
from juliacall import Main as jl

# Load the Julia GraphStoch module once when this file is imported.
jl.include("../julia/GraphStoch/src/GraphStoch.jl")
_GraphStoch = jl.GraphStoch


class GraphSDE:
    """
    Stochastic Differential Equation solver on graph topology.

    Wraps the Julia GraphStoch engine so it can be used directly
    from Python with NumPy arrays.
    """

    def __init__(self, adjacency, noise_level=0.5, dt=0.1):
        self.adjacency = np.asarray(adjacency, dtype=float)
        self.noise_level = noise_level
        self.dt = dt
        self.laplacian = _GraphStoch.graph_laplacian(self.adjacency)

    def simulate(self, initial_state, time_steps=100):
        X0 = np.asarray(initial_state, dtype=float)
        history = _GraphStoch.sde_solve(
            self.laplacian, X0, self.dt, time_steps, self.noise_level
        )
        return np.array([np.array(x) for x in history])

    def denoise(self, noisy_state, time_steps=50):
        X0 = np.asarray(noisy_state, dtype=float)
        history = _GraphStoch.diffuse(self.laplacian, X0, self.dt, time_steps)
        return np.array(history[-1])

    def stable_dt(self):
        eigenvalues = np.linalg.eigvalsh(np.array(self.laplacian))
        lambda_max = eigenvalues[-1]
        return 2.0 / lambda_max
    def denoise_with_reaction(self, noisy_state, beta, time_steps=50, sigma=0.0):
        X0 = np.asarray(noisy_state, dtype=float)
        result = _GraphStoch.diffuse_with_reaction(
            X0, self.laplacian, beta, sigma, self.dt, time_steps
        )
        return np.array(result)