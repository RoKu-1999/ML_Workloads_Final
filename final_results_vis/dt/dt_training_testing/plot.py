import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('train_test.csv')
df['train_time'] = df['train_time'] / 1e9
df['test_time'] = df['test_time'] / 1e9

# Map sgx values
df['sgx'] = df['sgx'].replace({0: 'Native', 1: 'SGX'})

# Initialize a 2x2 grid of plots
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(24, 12))

for i, dataset in enumerate(df['dataset'].unique()):
    # Filter the dataframe for current feature
    dataset_df = df[df['dataset'] == dataset]

    # Plot for train time vs sgx in the first row
    ax_train = sns.barplot(x='sgx', y='train_time', data=dataset_df, ax=axs[0, i], order=['Native', 'SGX'])
    axs[0, i].set_title(f"Dataset: {dataset} - Train Time", fontsize=18)
    axs[0, i].set_ylabel("Train Time", fontsize=18)

    # Calculate overhead for train time and annotate
    native_train_time = dataset_df.loc[dataset_df['sgx'] == 'Native', 'train_time'].values[0]
    sgx_train_time = dataset_df.loc[dataset_df['sgx'] == 'SGX', 'train_time'].values[0]
    overhead_train = sgx_train_time / native_train_time - 1.0
    mid_point_train = (sgx_train_time + native_train_time) / 3
    ax_train.text(1, mid_point_train, f'Overhead: {overhead_train:.2f}x', 
              ha='center', va='center', color='black', 
              weight='bold', fontsize=18)  # You can adjust the fontsize value as needed

    # Plot for test time vs sgx in the second row
    ax_test = sns.barplot(x='sgx', y='test_time', data=dataset_df, ax=axs[1, i], order=['Native', 'SGX'])
    axs[1, i].set_title(f"Dataset: {dataset} - Test Time", fontsize=18)
    axs[1, i].set_ylabel("Test Time", fontsize=18)

    # Calculate overhead for test time and annotate
    native_test_time = dataset_df.loc[dataset_df['sgx'] == 'Native', 'test_time'].values[0]
    sgx_test_time = dataset_df.loc[dataset_df['sgx'] == 'SGX', 'test_time'].values[0]
    overhead_test = sgx_test_time / native_test_time - 1.0
    mid_point_test = (sgx_test_time + native_test_time) / 3
    ax_test.text(1, mid_point_test, f'Overhead: {overhead_test:.2f}x', 
             ha='center', va='center', color='black', 
             weight='bold', fontsize=18)  # Adjust the fontsize as needed

# Adjust the layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("dt_train_test.pdf")
