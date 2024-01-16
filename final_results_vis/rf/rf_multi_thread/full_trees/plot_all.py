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
				overheads.append("+{:.3f}".format(bar.get_height() / heights[j] - 1))
				j = j + 1
			ax.bar_label(bars, labels=overheads, label_type='edge')
		i = i + 1
# Sample data frame to mimic the user's CSV data structure.
# This will be replaced with the actual data loading from the CSV file provided by the user.
df = pd.read_csv("result_rf.csv")
df['train_time'] = df['train_time'] / 1e9
df['test_time'] = df['test_time'] / 1e9
df['sgx'] = df['sgx'].replace({0: 'Native', 1: 'SGX'})


grouped_df = df.groupby(['model', 'sgx', 'threads'])[['train_time', 'test_time']].max().reset_index()

print(grouped_df)
    
plt.figure()
# Plotting
train = sns.barplot(data=grouped_df, x='threads', y='train_time', hue='sgx', hue_order=['Native', 'SGX'])
train.set_xlabel('Threads', fontsize=34)
train.set_ylabel('Train Time (seconds)', fontsize=34)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

label_overheads(train)

plt.savefig('rf_train_128_multithreaded.pdf')

plt.figure()

test = sns.barplot(data=grouped_df, x='threads', y='test_time', hue='sgx', hue_order=['Native', 'SGX'])
test.set_xlabel('Threads')
test.set_ylabel('Test Time (seconds)')

label_overheads(test)

plt.savefig('rf_test_128_multithreaded.png')
