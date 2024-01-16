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
				ovh = bar.get_height() / heights[j] - 1
				if ovh > 0:
					overheads.append("+{:.3f}".format(ovh))
				else:
					overheads.append("{:.3f}".format(ovh))
				j = j + 1
			ax.bar_label(bars, labels=overheads, label_type='center', fontsize=18)
		i = i + 1

df = pd.read_csv("dt_max_depth.csv")
df['max_depth'].fillna('None', inplace=True)
df['sgx'] = df['sgx'].replace({0: 'Native', 1: 'SGX'})
df['max_depth'] = df['max_depth'].astype('category')
df["train_time"] = df["train_time"]/1e9
df["test_time"] = df["test_time"]/1e9
print(df['max_depth'])
df = df.rename(columns={'sgx': 'Environment'})

sns.set_context("talk")

# Plotting
plt.figure(figsize=(24, 12))
hue_order = ['Native', 'SGX']
ax = sns.barplot(data=df, x="max_depth", y="train_time", hue="Environment", hue_order=hue_order)
label_overheads(ax)
plt.xlabel("Maximum Tree Depth", fontsize=20)
plt.ylabel("Training Time", fontsize=20)
ax.tick_params(axis='x', labelsize=18)  # Increase x-axis tick label size
ax.tick_params(axis='y', labelsize=18)  # Increase y-axis tick label size
ax.legend(fontsize='xx-large')  # Adjust the fontsize as needed
plt.tight_layout()

# Save the figure
plt.savefig("dt_train_time_max_depth.pdf")

# Plotting
plt.figure(figsize=(24, 12))
ax = sns.barplot(data=df, x="max_depth", y="test_time", hue="Environment")
label_overheads(ax)
plt.xlabel("Maximum Tree Depth", fontsize=20)
plt.ylabel("Test Time", fontsize=20)  # Adjusted label from "Training Time" to "Test Time"
ax.tick_params(axis='x', labelsize=18)  # Increase x-axis tick label size
ax.tick_params(axis='y', labelsize=18)  # Increase y-axis tick label size
ax.legend(fontsize='xx-large')  # Adjust the fontsize as needed
plt.tight_layout()

# Save the figure
plt.savefig("dt_test_time_max_depth.pdf")
