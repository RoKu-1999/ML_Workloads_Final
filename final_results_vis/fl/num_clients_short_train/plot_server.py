import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("server_result.txt")

# Plotting with seaborn and matplotlib
sns.barplot(data=data, x='num_clients', y='server_time_fl', hue='sgx')
plt.title('Training Time vs Number of Clients')
plt.xlabel('Number of Clients')
plt.ylabel('Training Time (fl)')
plt.grid(True)
plt.savefig("fl_server.png")