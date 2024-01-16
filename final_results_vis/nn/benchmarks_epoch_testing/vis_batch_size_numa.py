import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
			ax.bar_label(bars, labels=overheads, label_type='center')
		i = i + 1

# Load your data
df = pd.read_csv('result_epoch_eval.txt')

# Normalize the 'epoch_train' by dividing by 1e9 if necessary
df['epoch_test'] = df['epoch_test'] / 1e9

# Filter the DataFrame for rows where sgx == 1
sgx_one_df = df[df['sgx'] == 1]

# Create a FacetGrid for each batch size
g = sns.FacetGrid(sgx_one_df, col="batch_size", height=4, aspect=1, sharey=False)

# Map the lineplot to the FacetGrid
g.map_dataframe(sns.lineplot, x="threads", y="epoch_test", hue="arch")

# Add titles and labels
g.set_titles("Batch Size: {col_name}")
g.set_axis_labels("Number of Threads", "Average Training Time (in billions of epochs)")

# Adjust the layout to make room for the legend
plt.tight_layout()

# Add legend outside of the plots
g.add_legend(title="Architecture", bbox_to_anchor=(1.05, 0.5), loc='center left')

# Save the plot
plt.savefig("nn_testing_numa.pdf", bbox_inches="tight")

# Show the plot
plt.show()





