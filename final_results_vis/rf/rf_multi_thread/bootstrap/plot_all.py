import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
train.set_xlabel('Threads')
train.set_ylabel('Train Time (seconds)')

plt.savefig('rf_train_multithreaded_bootstrap.pdf')

plt.figure()

test = sns.barplot(data=grouped_df, x='threads', y='test_time', hue='sgx', hue_order=['Native', 'SGX'])
test.set_xlabel('Threads')
test.set_ylabel('Test Time (seconds)')

plt.savefig('rf_test_multithreaded_bootstrap.png')
