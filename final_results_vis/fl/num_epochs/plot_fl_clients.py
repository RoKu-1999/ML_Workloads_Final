import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('FL_Train_Clients.csv')

plt.rcParams["figure.figsize"] = (10, 5)

# epochs,num_clients,batch_size,sgx,ssl,median,mean,max,min,train_test_time,fl_time
df2 = df[["num_clients", "train_test_time", "mean", "sgx", "fl_time"]]
print(df2)



ax = sns.lineplot(data=df2, x="num_clients", y="fl_time", hue="sgx")

# Set label for x-axis
ax.set_xlabel( "Number of Clients" , size = 12 )
  
# Set label for y-axis
ax.set_ylabel( "Mean FL Time" , size = 12 )
  
# Set title for plot
ax.set_title( "FL Overall Time according to Client Increase" , size = 18 )

plt.savefig("fl_overall_clients.png")



ax = sns.lineplot(data=df2, x="num_clients", y="train_test_time")

# Set label for x-axis
ax.set_xlabel( "Number of Clients" , size = 12 )
  
# Set label for y-axis
ax.set_ylabel( "Train Test Time" , size = 12 )
  
# Set title for plot
ax.set_title( "FL Training Time according to Client Increase" , size = 18 )

plt.savefig("fl_train_clients.png")



ax = sns.lineplot(data=df2, x="num_clients", y="mean", hue="sgx")

# Set label for x-axis
ax.set_xlabel( "Number of Clients" , size = 12 )
  
# Set label for y-axis
ax.set_ylabel( "Mean Communication Time" , size = 12 )
  
# Set title for plot
ax.set_title( "FL Communication Time according to Client Increase" , size = 18 )

plt.savefig("fl_comm_clients.png")