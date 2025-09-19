import numpy as np
import matplotlib.pyplot as plt

# Define x range
x = np.linspace(0, 200, 200)

# Define lambda values (10^a, with a from 0 to 4 in steps of 0.5)
a_values = np.arange(0, 4.5, 0.5)  # 0, 0.5, 1.0, ..., 4.0
lambda_values = 10**-a_values
print(lambda_values)

# Plot each curve
plt.figure(figsize=(10, 6))
for lam in lambda_values:
    y = np.exp(-lam * x)
    print(y[10])
    plt.plot(x, y, label=f"$\lambda=10^{{{int(np.log10(lam))}}}$" if lam in [10**i for i in range(5)] else f"{lam:.1e}")

# Add labels, title, legend
plt.xlabel("t", fontsize=12)
plt.ylabel("$\mu=e^{-\\lambda t}$", fontsize=12)
plt.title("Plasticity Decay for Different $\lambda$ Values", fontsize=12)
# plt.yscale("log")  # Log scale for clarity
# plt.legend()
# plt.grid(True, which="both", ls="--", alpha=0.6)

# plt.show()

# Set labels and title with font sizes
# plt.xlabel('t', fontsize=14)
# plt.ylabel(r'$\mu_t$', fontsize=14)
# plt.title(r'Neuromodulation term ($\mu_t$) updated through time (t)', fontsize=16)

# Set legend font size
plt.legend(fontsize=12)

# Optional: Adjust tick label font sizes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.grid(True)
plt.tight_layout()
plt.show()