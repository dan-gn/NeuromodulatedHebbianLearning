import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define x range
x = np.linspace(0, 200, 200)

# Define lambda values (10^a, with a from 0 to 4 in steps of 0.5)
a_values = np.arange(0, 4.5, 0.5)  # 0, 0.5, 1.0, ..., 4.0
lambda_values = 10**-a_values
print(lambda_values)

# colors = sns.color_palette('tab20')

colors = [
    '#4E79A7', '#F28E2B', '#E15759',
    '#76B7B2', '#59A14F', '#EDC948',
    '#B07AA1', '#FF9DA7', '#9C755F'
]

# colors = [
#     '#74a9cf',
#     '#4a90c2',
#     '#2b7bba',
#     '#1f6aa5',
#     '#175a8c',
#     '#0f4c75',
#     '#0b3c5d',
#     '#082c45',
#     '#051c2c'
# ]
print(len(colors))
# Plot each curve
plt.figure(figsize=(10, 6))
# for lam in lambda_values:
#     y = np.exp(-lam * x)
#     print(y[10])
#     # plt.plot(x, y, label=fr"$10^{{{int(np.log10(lam))}}}$" if lam in [10**i for i in range(5)] else f"{lam:.1e}")
#     plt.plot(x, y, label=fr"$10^{{{int(np.log10(lam))}}}$" if lam in [10**i for i in range(5)] else f"{lam:.1e}")

for i, exponent in enumerate(a_values):
    lambda_value = 10 ** -exponent
    y = np.exp(-lambda_value * x)
    label = f'$\lambda = 10^{{{-exponent}}}$'
    if exponent == 0:
        label = f'$\lambda = 1$'
    sns.lineplot(
    x=x,
    y=y,
    label=label,
    linewidth=2,
    color=colors[i]
    )

plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), frameon=False)

# Add labels, title, legend
plt.xlabel("t", fontsize=14)
plt.ylabel("$\mu=e^{-\\lambda t}$", fontsize=14)
plt.title("Plasticity Decay for Different $\lambda$ Values", fontsize=14)
# plt.yscale("log")  # Log scale for clarity
# plt.legend()
# plt.grid(True, which="both", ls="--", alpha=0.6)

# plt.show()

# Set labels and title with font sizes
# plt.xlabel('t', fontsize=14)
# plt.ylabel(r'$\mu_t$', fontsize=14)
# plt.title(r'Neuromodulation term ($\mu_t$) updated through time (t)', fontsize=16)

# Set legend font size
# plt.legend(fontsize=12)

# Optional: Adjust tick label font sizes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.grid(True)
plt.tight_layout()
plt.show()