import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data provided by the user

# Convert the sample data into a DataFrame
df = pd.read_csv('result_epoch_eval.txt')

df['epoch_test'] = df['epoch_test'] / 1e9

# Group by the combination of sgx, threads, arch, and batch_size and sum the epoch_train values
grouped_df = df.groupby(['sgx', 'threads', 'arch', 'batch_size']).min().reset_index()

print(grouped_df)

# Plotting
for arch in grouped_df['arch'].unique():
    for batch_size in grouped_df['batch_size'].unique():
        # Subsample df per arch and batch_size
        subsample = grouped_df[(grouped_df['arch'] == arch) & (grouped_df['batch_size'] == batch_size)]

        plt.figure()
        sns.barplot(data=subsample, x='threads', y='epoch_test', hue='sgx')
        plt.title('Neural Network Evaluation')
        plt.xlabel('Number of Threads')
        plt.ylabel('Eval Time')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("nn_test_"+str(arch)+"_"+str(batch_size)+".png")
