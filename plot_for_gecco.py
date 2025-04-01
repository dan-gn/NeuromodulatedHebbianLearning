import numpy as np
import matplotlib.pyplot as plt

# Define t and y
t = np.linspace(0, 600, 1000)
y = np.exp(-0.01 * t)

threshold = 0.04

# Split into two conditions: y > 0.1 (blue) and y <= 0.1 (red)
above_threshold = y > threshold
below_threshold = ~above_threshold

# Create the plot
plt.figure(figsize=(8, 5))

# Plot blue part
plt.plot(t[above_threshold], y[above_threshold], color='blue', label='Plasticity')

# Plot red part
plt.plot(t[below_threshold], y[below_threshold], color='red', label='Almost static')


# Add threshold line for clarity (optional)
plt.axhline(threshold, color='gray', linestyle='--')

plt.xlabel('t')
plt.ylabel(r'$\mu_t$')
plt.title(r'Neuromodulation term ($\mu_t$) updated through time (t)')
plt.legend()
plt.grid(True)
plt.show()