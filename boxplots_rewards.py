import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# environment = 'CartPole-v1'
# environment = 'Acrobot-v1'
environment = 'MountainCar-v0'

# metric = 'best_score'
# metric = 'evaluation_score'
# metric = 'testing'
metric = 'n_iterations'

# 1. Read the CSV file into a dataframe
df = pd.read_csv("Experiments/Results/test_aics/experiments_log_100tries_v1.csv")
# df = pd.read_csv("final_results_gecco_april2.csv")

# 2. Filter the "environment" column for "CartPole-v1"
df_filtered = df[df["environment"] == environment]

# 3. Further filter the "model" column for "neuromodulated_hb"
# df_filtered = df_filtered[df_filtered["model"] == "neuromodulated_hb"]
df_filtered.loc[df_filtered["model"] == "abcd", "lambda_exp"] = -1
df_filtered.loc[df_filtered["model"] == "static", "lambda_exp"] = -2

df_filtered.loc[df_filtered["model"] == "neuromodulated_hb", "model"] = 'tm-HL'
df_filtered.loc[df_filtered["model"] == "abcd", "model"] = 'sHL'
df_filtered.loc[df_filtered["model"] == "static", "model"] = 'Fixed-ANN'

# 4. Make boxplots according to the "lambda_exp" column
# df_filtered.boxplot(column=metric, by=["lambda_exp"], figsize=(10, 6))  
# # Replace "some_metric" with the name of the column you want to plot values for

# plt.title("Boxplot of Some Metric by lambda_exp")
# plt.suptitle("")  # removes the automatic 'Boxplot grouped by lambda_exp' title
# plt.xlabel("lambda_exp")
# plt.ylabel("Testing fitness")
# plt.grid(False)
# plt.show()

# ---- Compute means ----
mean_table = (
    df_filtered.groupby(["lambda_exp", "model"])[metric]  # replace with your metric
    .mean()
    .reset_index()
)

print("\nMean values of 'some_metric' by lambda_exp and model:")
print(mean_table)

# ---- Compute medians ----
median_table = (
    df_filtered.groupby(["lambda_exp", "model"])[metric]  # replace with your metric
    .std()
    .reset_index()
)

print("\nMedian values of 'some_metric' by lambda_exp and model:")
print(median_table)

# Plot with seaborn (colored by model)
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df_filtered,
    x="lambda_exp",
    y=metric,      # replace with your column to plot
    hue="model",          # different colors for models
    palette="Set2"        # you can try "Set1", "Dark2", "Pastel1", "tab10", etc.
)

plt.title(f'Total reward of each model on the testing set of seeds for {environment[:-3]}', fontsize=12)
# plt.xlabel("$\lambda$", fontsize=12)
plt.xlabel(" ", fontsize=12)
plt.ylabel("Total reward", fontsize=12)
plt.legend(title="Model", fontsize=12)

# Change tick labels
# labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
# labels = [x/2 for x in range(9)]
# labels = [str(-(x/2)) for x in range(9)]
# labels = [['^' + y for y in x] for x in labels]
# print(labels)
# labels = ['^-^0', '^-^0^.^5']
# labels = [f'tm-HL $\lambda=10^[-0.5]$' for x in labels]
labels = [r"$10^{-0}$", r"$10^{-0.5}$"]
labels += [r"$10^{-1.0}$", r"$10^{-1.5}$"]
labels += [r"$10^{-2.0}$", r"$10^{-2.5}$"]
labels += [r"$10^{-3.0}$", r"$10^{-3.5}$"]
labels += [r"$10^{-4.0}$"]
labels = [f'tm-HL $\lambda=${x}' for x in labels]
labels = ['Fixed-ANN', 'sHL'] + labels

# plt.xticks(range(2, len(labels)+2), labels)
plt.xticks(range(0, len(labels)), labels, rotation=45, fontsize=12)
plt.yticks(fontsize=12)

plt.grid(False)
plt.tight_layout()
plt.show()