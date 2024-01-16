import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (10, 5)

df = pd.read_csv('Training_FL_Epochs.csv')

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
				overheads.append("{:.3f}".format(bar.get_height() / heights[j]))
				j = j + 1
			ax.bar_label(bars, labels=overheads, label_type='center', fontsize=26)
		i = i + 1

# epochs,num_clients,batch_size,sgx,ssl,median,mean,max,min,train_test_time,fl_time
df2 = df[["train_test_time", "mean", "sgx", "fl_time", "epochs"]]
df2['sgx'] = df2['sgx'].replace({0: 'Native', 1: 'SGX'})
print(df2)

plt.figure(figsize=(24, 12))

ax = sns.barplot(data=df2, x="epochs", y="mean", hue="sgx")

label_overheads(ax)

# Set label for x-axis
ax.set_xlabel( "Epochs" , fontsize=34)

# Set label for y-axis
ax.set_ylabel( "Mean Communication Time" , fontsize=34)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
ax.tick_params(axis='x', labelsize=26)  # Increase x-axis tick label size
ax.tick_params(axis='y', labelsize=26)  # Increase y-axis tick label size
ax.legend(fontsize='xx-large')  # Adjust the fontsize as needed

plt.savefig("fl_comm_epochs.pdf")