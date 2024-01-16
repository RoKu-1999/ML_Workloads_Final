import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Convert the data to a DataFrame
df = pd.read_csv("client_result_agg.txt")

# Replace sgx values with corresponding labels
df['sgx'] = df['sgx'].map({0: 'Native', 1: 'SGX'})

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x='epochs', y='mean', hue='sgx', data=df)
plt.title('Mean Communication Time Across Several Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Communication Time')
plt.legend(title='SGX')
plt.savefig("epochs.png")
