import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data frame to mimic the user's CSV data structure.
# This will be replaced with the actual data loading from the CSV file provided by the user.
df = pd.read_csv("result_rf.csv")
df['train_time'] = df['train_time'] / 1e9
df['test_time'] = df['test_time'] / 1e9
filtered_df = df[df['threads'] >= 32]


grouped_df = filtered_df.groupby(['model', 'sgx', 'threads'])[['train_time', 'test_time']].max().reset_index()

print(grouped_df)
    
plt.figure()
# Plotting
barplot = sns.barplot(data=grouped_df, x='threads', y='train_time', hue='sgx')
barplot.set_xlabel('Threads')
barplot.set_ylabel('Train Time (seconds)')
barplot.set_title('Train Time vs. Threads by SGX')

plt.savefig('rf_train_128_numa.png')

plt.figure()

barplot = sns.barplot(data=grouped_df, x='threads', y='test_time', hue='sgx')
barplot.set_xlabel('Threads')
barplot.set_ylabel('Test Time (seconds)')
barplot.set_title('Test Time vs. Threads by SGX')

plt.savefig('rf_test_128_numa.png')
