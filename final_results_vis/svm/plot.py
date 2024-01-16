import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def label_overheads(ax):
	i = 0
	for bars in ax.containers:

		print(i)

		print(len(ax.containers))
		print(len(bars))
		y1 = bars[-2].get_height()
		y2 = bars[-1].get_height()

		overhead = y1 / y2

		print(overhead)
		if i == 0:
			heights = []
			for bar in bars:
				heights.append(bar.get_height())
		if i == 1:
			j = 0
			overheads = []
			for bar in bars:
				overheads.append("+{:.3f}".format(bar.get_height() / heights[j] - 1))
				j = j + 1
			ax.bar_label(bars, labels=overheads, label_type='center')
		i = i + 1

# Reading the data
df = pd.read_csv("result_svm.csv")
df["train_time"] = df["train_time"] / 1e9
df["test_time"] = df["test_time"] / 1e9
df['sgx'] = df['sgx'].replace({0: 'Native', 1: 'SGX'})
df = df.rename(columns={'sgx': 'Environment'})

# Plotting setup
hue_order = ['Native', 'SGX']
plt.figure(figsize=(24, 12))

# First subplot for train_time
plt.subplot(1, 2, 1)
ax1 = sns.barplot(data=df, x="num_samples", y="train_time", hue="Environment", hue_order=hue_order)
label_overheads(ax1)
plt.xlabel("Number of Datapoints", fontsize=18)
plt.ylabel("Training Time (seconds)", fontsize=18)
ax1.tick_params(axis='x', labelsize=18)  # Increase x-axis tick label size
ax1.tick_params(axis='y', labelsize=18)  # Increase y-axis tick label size

# Second subplot for test_time
plt.subplot(1, 2, 2)
ax2 = sns.barplot(data=df, x="num_samples", y="test_time", hue="Environment", hue_order=hue_order)
label_overheads(ax2)
plt.xlabel("Number of Datapoints", fontsize=20)
plt.ylabel("Testing Time (seconds)", fontsize=20)
ax2.tick_params(axis='x', labelsize=18)  # Increase x-axis tick label size
ax2.tick_params(axis='y', labelsize=18)  # Increase y-axis tick label size

# General layout adjustments
plt.tight_layout()

# Save the figure
plt.savefig("train_test_svm_comparison.pdf")
