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
			ax.bar_label(bars, labels=overheads, label_type='center')
		i = i + 1

df = pd.read_csv("result_rf.csv")
df['max_depth'].fillna('None', inplace=True)
df['sgx'] = df['sgx'].replace({0: 'Native', 1: 'SGX'})
df['max_depth'] = df['max_depth'].astype('category')
df["train_time"] = df["train_time"]/1e9
df["test_time"] = df["test_time"]/1e9
df["train_rows"] = df["train_rows"]/11
df["test_rows"] = df["test_rows"]/11
print(df['max_depth'])
df = df.rename(columns={'sgx': 'Environment'})

sns.set_context("talk")

# Plotting
plt.figure(figsize=(24, 12))
hue_order = ['Native', 'SGX']
ax = sns.barplot(data=df, x="train_rows", y="train_time", hue="Environment", hue_order=hue_order)
label_overheads(ax)
plt.title("Distribution of Train Time across different Tree Depths", fontsize=20)
plt.xlabel("Maximum Tree Depth", fontsize=18)
plt.ylabel("Training Time", fontsize=18)
plt.tight_layout()

# Save the figure
plt.savefig("train_rf_none.png")

# Plotting
plt.figure(figsize=(24, 12))
ax = sns.barplot(data=df, x="test_rows", y="test_time", hue="Environment")
label_overheads(ax)
plt.title("Distribution of Test Time  across different Tree Depths", fontsize=20)
plt.xlabel("Maximum Tree Depth", fontsize=18)
plt.ylabel("Test Time", fontsize=18)  # Adjusted label from "Training Time" to "Test Time"
plt.tight_layout()

# Save the figure
plt.savefig("test_rf_none.png")
