import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42  # Use Type 42 (TrueType) fonts
mpl.rcParams['ps.fonttype'] = 42   # For saving as .ps if needed

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

# plt.xlabel('t')
# plt.ylabel(r'$\mu_t$')
# plt.title(r'Neuromodulation term ($\mu_t$) updated through time (t)')
# plt.legend()
# plt.grid(True)
# plt.show()

# Set labels and title with font sizes
plt.xlabel('t', fontsize=14)
plt.ylabel(r'$\mu_t$', fontsize=14)
plt.title(r'Neuromodulation term ($\mu_t$) updated through time (t)', fontsize=16)

# Set legend font size
plt.legend(fontsize=14)

# Optional: Adjust tick label font sizes
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.grid(True)
plt.tight_layout()
plt.show()