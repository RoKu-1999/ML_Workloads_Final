import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("client_result.csv")
data['sgx'] = data['sgx'].replace({0: 'Native', 1: 'SGX'})

# Plotting with seaborn and matplotlib
sns.lineplot(data=data, x='num_clients', y='median', hue='sgx')
plt.xlabel('Number of Clients')
plt.ylabel('Cumulative Communication Time with Server')
plt.grid(True)
plt.axvline(x=32, color='red', linestyle=':')
plt.savefig("fl_clients.pdf")