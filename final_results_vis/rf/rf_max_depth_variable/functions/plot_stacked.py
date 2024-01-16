import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV data
df = pd.read_csv('top_10_functions.csv')

# Group by both function and sgx, then reset the index to flatten the DataFrame
df = df.groupby(['function', 'sgx', 'max_depth'])['percent_spent'].sum().reset_index()

print(df)
# Filter the data to only include rows where percent_spent is at least 0.01
df = df[df['percent_spent'] >= 0.01]

# Prepare the data for stacked bar chart
pivot_df = df.pivot_table(index='max_depth', columns=['sgx', 'function'], values='percent_spent', aggfunc='sum').fillna(0)

# Find common functions between both sgx=0 and sgx=1
common_functions = set(pivot_df[0].columns).intersection(set(pivot_df[1].columns))

# Sort common functions based on combined percent_spent
sorted_common_functions = df[df['function'].isin(common_functions)].groupby('function')['percent_spent'].sum().sort_values(ascending=False).index.tolist()

# Find additional functions for sgx=0 and sgx=1
additional_functions_sgx_0 = set(pivot_df[0].columns) - common_functions
additional_functions_sgx_1 = set(pivot_df[1].columns) - common_functions

# Combine sorted function lists
sorted_functions = sorted_common_functions + list(additional_functions_sgx_0) + list(additional_functions_sgx_1)

# Reorder columns in pivot_df according to sorted functions
ordered_columns = [(sgx, function) for function in sorted_functions for sgx in [1, 0] if (sgx, function) in pivot_df.columns]
pivot_df = pivot_df[ordered_columns]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 16), sharey=True)

# Create a categorical colormap based on the unique functions
colors = plt.cm.tab20c(np.linspace(0, 1, len(sorted_functions)))
color_map = dict(zip(sorted_functions, colors))

for idx, sgx_value in enumerate([0, 1]):
    # Plot with color map
    pivot_df[sgx_value].plot(kind='bar', stacked=True, ax=axes[idx], color=[color_map[func] for func in pivot_df[sgx_value].columns], legend=False)
    
    if sgx_value == 0:
        axes[idx].set_title(f"Training without Intel SGX", fontsize=36)
    else:
        axes[idx].set_title(f"Training inside Intel SGX", fontsize=36)
        
    axes[idx].set_ylabel("Time spent in Functions in Percent", fontsize=32)
    
    # Adjust tick label sizes
    axes[idx].tick_params(axis='both', labelsize=30)
    axes[idx].tick_params(labelrotation=0, labelsize=30)

# Set x-label for both subplots with adjusted size
axes[0].set_xlabel("Tree Max Depth", fontsize=34)
axes[1].set_xlabel("Tree Max Depth", fontsize=34)

# Add a shared legend outside of the plots
fig.legend([plt.Rectangle((0,0),1,1, fc=color_map[func]) for func in reversed(sorted_functions)], list(reversed(sorted_functions)), loc='center right', fontsize='xx-large')

fig.subplots_adjust(right=0.7, wspace=0.3)  # Adjust spacing between subplots and right boundary to account for legend

plt.savefig("rfs_depth_CPU_functions_stacked.pdf")
