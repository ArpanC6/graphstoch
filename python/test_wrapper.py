import numpy as np
from graphstoch import GraphSDE

A = np.array([[0.0, 1.0, 0.0, 0.0],
              [1.0, 0.0, 1.0, 0.0],
              [0.0, 1.0, 0.0, 1.0],
              [0.0, 0.0, 1.0, 0.0]])

model = GraphSDE(A, noise_level=0.5, dt=0.1)

print("Laplacian:")
print(model.laplacian)

print("\nSuggested stable dt:")
print(model.stable_dt())

X0 = np.array([10.0, 0.0, 0.0, 0.0])

print("\nSimulate (with noise), 5 steps:")
result = model.simulate(X0, time_steps=5)
print(result)

print("\nDenoise, 5 steps:")
denoised = model.denoise(X0, time_steps=5)
print(denoised)
