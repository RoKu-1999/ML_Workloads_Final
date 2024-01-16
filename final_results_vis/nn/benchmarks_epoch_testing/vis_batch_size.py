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

# Normalize the 'epoch_test' by dividing by 1e9 if necessary
df['epoch_test'] = df['epoch_test'] / 1e9
df['sgx'] = df['sgx'].replace({0: 'Native', 1: 'SGX'})

# Filter the DataFrame for rows where threads == 1
threads_one_df = df[df['threads'] == 1]

# Create a FacetGrid
g = sns.FacetGrid(threads_one_df, col="batch_size", height=4, aspect=1)

# Map the barplot to the FacetGrid
g.map_dataframe(sns.barplot, x="arch", y="epoch_test", hue="sgx", hue_order=['Native','SGX'])

# Add titles and labels
g.set_titles("Batch Size: {col_name}")
g.set_axis_labels("Architecture", "Average Inference Time")

# Adjust the layout to make room for the legend
plt.tight_layout()

# Add legend outside of the plots
g.add_legend(title="SGX", bbox_to_anchor=(1.05, 0.5), loc='center left')

# Save the plot
plt.savefig("nn_inference_batch_sizes.pdf", bbox_inches="tight")

# Show the plot
plt.show()




