import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("client_result.csv")

# Plotting with seaborn and matplotlib
sns.lineplot(data=data, x='num_clients', y='train_test_time', hue='sgx')
plt.title('Communication Time with Server')
plt.xlabel('Number of Clients')
plt.ylabel('Commulated Communication Time with Server')
plt.grid(True)
plt.savefig("fl_clients_fl_time.png")