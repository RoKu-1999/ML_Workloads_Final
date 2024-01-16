import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the datasets
data1 = pd.read_csv('cpu-times.csv')
data2 = pd.read_csv('top_10_functions.csv')

# Add the percent_train column
#data1['percent_train'] = [(i) * 10 % 90 + 10 for i in range(len(data1))]
#data1['num_rows'] = data1['num_rows'] / 110000
#data1['num_rows'] = data1['num_rows'].astype(int)
#data1 = data1.rename(columns={'num_rows': 'percent_train'})

# Set the columns 'sgx' and 'percent_train' as indices for both dataframes before the join
data1.set_index(['sgx', 'num_trees'], inplace=True)
data2.set_index(['sgx', 'num_trees'], inplace=True)

# Perform the join
joined_data = data1.join(data2, how='inner')  # inner join by default, you can change it to 'left', 'right', or 'outer' as needed.
joined_data.reset_index(inplace=True)
print(joined_data)
filtered_df = joined_data[joined_data['num_trees'].isin([1, 64])]
# Reset index if you want

print(filtered_df)

# Compute time_in_func from cpu_time and percent_spent
filtered_df['time_in_func'] = filtered_df['cpu_time'] * filtered_df['percent_spent']

filtered_df['sgx'] = filtered_df['sgx'].replace({0: 'Native', 1: 'SGX'})
# Combine sgx and percent_train columns
filtered_df['sgx_trees'] = filtered_df['sgx'].astype(str) + '-' + filtered_df['num_trees'].astype(str)

vis = pd.DataFrame()
vis['sgx_trees'] = filtered_df['sgx_trees']
vis['time_in_func'] = filtered_df['time_in_func']
vis['function'] = filtered_df['function']

# Pivot the DataFrame to get the functions as columns and their respective times as values

#functions = ['tokenize_bytes', 'splitter_introsort', 'sgx_copy_to_enclave_verified', 'precise_xstrtod', 'node_split_best', 'memset', 'memcpy', 'mbedtls_internal_sha256_process', 'management8internal20computeSumAVX512ImplIdEET', 'management8internal11splitColumnIdiLNS', 'kernel_func', 'get_trusted_or_allowed_file', 'Z11aquicksort', 'PyArray_Nonzero', 'DensePartitioner_sort_samples_and_feature_values', 'DOUBLE_pairwise_sum', 'ClassificationCriterion_update', 'ClassificationCriterion_init', 'AVX512_SKX_isnan_DOUBLE']

#gramine_functions = ['get_trusted_or_allowed_file','mbedtls_internal_sha256_process', 'memcpy', 'memset', 'sgx_copy_to_enclave_verified']
#kernel_function = 'kernel_func'

#sorted_functions = [func for func in functions if func not in gramine_functions and func != kernel_function] + \
#                   [kernel_function] + \
#                   [func for func in gramine_functions if func in functions]



# Aggregating data before pivot
df_agg = vis.groupby(['sgx_trees', 'function']).time_in_func.sum().reset_index()

# Pivot the aggregated data
df_pivot = df_agg.pivot(index='sgx_trees', columns='function', values='time_in_func').fillna(0)
#df_pivot = df_pivot[sorted_functions]
order = ['Native-1', 'SGX-1', 'Native-64', 'SGX-64']
df_pivot = df_pivot.reindex(order)
# Generate colors with decreasing saturation
#gramine_colors = sns.cubehelix_palette(len(gramine_functions), start=2, rot=0, dark=0.2, light=.8, reverse=True)
#kernel_color = ["grey"]
#sklearn_colors = sns.diverging_palette(220, 20, n=len(functions) - len(gramine_functions) - 1)  # -1 for kernel_func

# Create the color dictionary
#colors_dict = dict(zip(gramine_functions, gramine_colors))
#colors_dict[kernel_function] = kernel_color[0]
#for func, color in zip([f for f in functions if f not in gramine_functions and f != kernel_function], sklearn_colors):
#    colors_dict[func] = color

# Extract colors in the order of sorted_functions for plotting
#colors_ordered = [colors_dict[func] for func in sorted_functions]

# Plot with the specified colors
ax = df_pivot.plot(kind='bar', stacked=True, figsize=(10, 6))#, color=colors_ordered)

# Setting labels and title
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_xlabel("SGX Trees")
ax.set_ylabel("Time in Func")
ax.set_title("Stacked Bar Chart of Time in Func by Execution Environment")
ax.legend(title="Function", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("cpu_functions_absolute.png")