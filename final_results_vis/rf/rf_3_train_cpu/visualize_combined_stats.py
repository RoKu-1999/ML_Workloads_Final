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

# Calculate total normalized values for RF and sort the events accordingly
sorted_events = df[df['model'] == 'rf'].groupby('event')['norm_counter_val'].sum().sort_values(ascending=False).index.tolist()
df = df[df['model'] == 'rf']
df['model'] = df['model'].replace({'dt': 'DT', 'rf': 'RF'})
# Plotting
plt.figure(figsize=(12, 6))
ax = sns.barplot(data=df, x='event', y='norm_counter_val', order=sorted_events)
label_overheads(ax)

plt.ylabel('Normalized Counter Value', fontsize=20)
plt.xlabel('Event Counter', fontsize=20)
plt.xticks(rotation=45)
ax.tick_params(axis='x', labelsize=18)  # Increase x-axis tick label size
ax.tick_params(axis='y', labelsize=18)  # Increase y-axis tick label size
plt.legend(title='Model Environment', title_fontsize=15, fontsize=13)
ax.legend(fontsize='xx-large')  # Adjust the fontsize as needed

plt.tight_layout()

# Save the figure
plt.savefig("Memory_Optimization_Events.pdf")
plt.close()
