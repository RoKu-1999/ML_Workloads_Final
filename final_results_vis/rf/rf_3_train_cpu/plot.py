import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def label_overheads(ax):
    i = 0
    for bars in ax.containers:
        y1 = bars[-2].get_height()
        y2 = bars[-1].get_height()
        overhead = y1 / y2
        if i == 0 or i == 2:
            heights = []
            for bar in bars:
                heights.append(bar.get_height())
        if i == 1 or i == 3:
            j = 0
            overheads = []
            for bar in bars:
                overheads.append("+{:.3f}".format(bar.get_height() / heights[j] - 1))
                j = j + 1
            ax.bar_label(bars, labels=overheads, label_type='edge', fontsize=24)
        i = i + 1

df_dt = pd.read_csv("dt_3_train.csv")
df_dt["Train_Time"] = df_dt["Train_Time"]/1e9
df_dt['sgx'] = df_dt['sgx'].replace({0: 'Native', 1: 'SGX'})
df_dt = df_dt.rename(columns={'sgx': 'Environment'})
df_dt['Model_Environment'] = df_dt['Model'] + '-' + df_dt['Environment']

df_rf = pd.read_csv("rf_3_train.csv")
df_rf["Train_Time"] = df_rf["Train_Time"]/1e9
df_rf['sgx'] = df_rf['sgx'].replace({0: 'Native', 1: 'SGX'})
df_rf = df_rf.rename(columns={'sgx': 'Environment'})
df_rf['Model_Environment'] = df_rf['Model'] + '-' + df_rf['Environment']

# Combining the dataframes for plotting
df_combined = pd.concat([df_dt, df_rf])
df_combined['Rows'] = df_combined['Rows']/110000+0.5
df_combined['Rows'] = df_combined['Rows'].astype(int)

hue_order = ['dt-Native','dt-SGX','rf-Native','rf-SGX']
# Plotting
plt.figure(figsize=(24, 12))
ax = sns.barplot(x='Rows', y='Train_Time', hue='Model_Environment', hue_order=hue_order, data=df_combined, estimator=np.mean, ci=None)

label_overheads(ax)
plt.ylabel('Train Time (in seconds)', fontsize=34)
plt.xlabel('Training Data Percent', fontsize=34)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
ax.tick_params(axis='x', labelsize=30)  # Increase x-axis tick label size
ax.tick_params(axis='y', labelsize=30)  # Increase y-axis tick label size
#plt.legend(title='Model Environment', title_fontsize=15, fontsize=13)
#ax.legend(fontsize='xx-large')  # Adjust the fontsize as needed

# Save the figure
plt.savefig("train_time_dt_rf_depth_3.pdf")