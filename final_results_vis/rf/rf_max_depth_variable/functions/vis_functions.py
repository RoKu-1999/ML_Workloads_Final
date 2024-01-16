import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Creating a DataFrame from the provided data
df = pd.read_csv('top_10_functions.csv')

print(df)

# Plotting the line plot
df["sgx_label"] = df["sgx"].apply(lambda x: f"SGX {x}")

print(df)

# Creating the seaborn plot
plt.figure(figsize=(10, 6))

fig, axes = plt.subplots(1, 5, figsize=(23, 5), sharey=True)

columns_plot = ['SGX 1', 'SGX 0']

for i, each in enumerate(columns_plot):
    sns.lineplot(data=df, ax = axes[i], x = "max_depth", y = "percent_spent", hue="function")



plt.xlabel("Max Depth")
plt.ylabel("Percent Spent")
plt.title("Seaborn Line Plot of Percent Spent by Function and SGX")
plt.legend(title="Function and SGX")
plt.grid(True)
plt.show()
