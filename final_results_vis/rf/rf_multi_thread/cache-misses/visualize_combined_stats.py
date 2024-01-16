import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def label_overheads(ax):
    # Iterate over the bars
    for bar in ax.patches:
        # Calculate 100 - height
        label = 100 - bar.get_height()

        # Get the position for the label
        x = bar.get_x() + bar.get_width() / 2  # Center of the bar
        y = bar.get_height() / 2  # Middle of the bar's height

        # Place the label in the middle of the bar
        ax.text(x, y, f'-{label:.1f}%', ha='center', va='center', fontsize=24)

# Read data
df = pd.read_csv("combined_stats.csv")

df['norm_counter_val'] = df.groupby('event')['counter_val'].transform(lambda x: (x / x.max()) * 100)
df = df[df['threads'] == 32]
print(df)

# Plotting
plt.figure(figsize=(12, 6))
ax = sns.barplot(data=df, x='event', y='norm_counter_val')
label_overheads(ax)
plt.ylabel('Normalized Counter Value', fontsize=20)
plt.xlabel('Event Counter', fontsize=20)
plt.xticks(rotation=45)
ax.tick_params(axis='x', labelsize=18)  # Increase x-axis tick label size
ax.tick_params(axis='y', labelsize=18)  # Increase y-axis tick label size
#legend = ax.get_legend()
#legend.set_title("Threads")
plt.tight_layout()
plt.show()

# Save the figure
plt.savefig("cache_misses_multithreaded.pdf")
plt.close()
