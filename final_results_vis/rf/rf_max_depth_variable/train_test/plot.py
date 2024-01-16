import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def label_overheads(ax):
	i = 0
	for bars in ax.containers:

		print(i)

		print(len(ax.containers))
		print(len(bars))

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
			ax.bar_label(bars, labels=overheads, label_type='edge')
		i = i + 1

df = pd.read_csv("result_rf.csv")
df['max_depth'].fillna(58, inplace=True)
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
ax = sns.lineplot(data=df, x="max_depth", y="train_time", hue="Environment", hue_order=hue_order)
#label_overheads(ax)
plt.xlabel("Maximum Tree Depth", fontsize=34)
plt.ylabel("Training Time", fontsize=34)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.tight_layout()

plt.savefig("rf_train_time_max_depth_line.pdf")

plt.figure(figsize=(24, 12))
hue_order = ['Native', 'SGX']
ax = sns.barplot(data=df, x="max_depth", y="train_time", hue="Environment", hue_order=hue_order)
label_overheads(ax)
plt.xlabel("Maximum Tree Depth", fontsize=25)
plt.ylabel("Training Time (in seconds)", fontsize=25)

plt.tight_layout()

# Save the figure
plt.savefig("rf_train_time_max_depth_bar.png")

# Plotting
plt.figure(figsize=(24, 12))
ax = sns.lineplot(data=df, x="max_depth", y="test_time", hue="Environment")
#label_overheads(ax)
plt.xlabel("Maximum Tree Depth", fontsize=25)
plt.ylabel("Test Time (in seconds)", fontsize=25)  # Adjusted label from "Training Time" to "Test Time"
plt.tight_layout()

# Save the figure
plt.savefig("rf_test_time_max_depth.pdf")
