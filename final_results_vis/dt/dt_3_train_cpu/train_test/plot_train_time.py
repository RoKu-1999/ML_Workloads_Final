import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

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
			ax.bar_label(bars, labels=overheads, label_type='center', fontsize=18)
		i = i + 1

df = pd.read_csv("dt_3_train.csv")
df["Train_Time"] = df["Train_Time"]/1e9
df['sgx'] = df['sgx'].replace({0: 'Native', 1: 'SGX'})
df = df.rename(columns={'sgx': 'Environment'})

# Plotting
hue_order = ['Native', 'SGX']
plt.figure(figsize=(24, 12))
ax = sns.barplot(data=df, x="MiB", y="Train_Time", hue="Environment", hue_order=hue_order)
label_overheads(ax)
plt.xlabel("Training Data in MiB", fontsize=34)
plt.ylabel("Training Time", fontsize=34)
ax.tick_params(axis='x', labelsize=30)  # Increase x-axis tick label size
ax.tick_params(axis='y', labelsize=30)  # Increase y-axis tick label size
ax.legend(fontsize='xx-large')  # Adjust the fontsize as needed

plt.tight_layout()

# Save the figure
plt.savefig("dt_3_train_time.pdf")
