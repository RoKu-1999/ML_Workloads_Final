import pandas as pd
import sys
import os

df = pd.read_csv("client_result_20.txt", header=None)
sum = 0
for i in range(7, len(df.columns)-1):
    sum = sum + df[i]


df2 = pd.DataFrame()
df2['client_id'] = df[0]
df2['epochs'] = df[1]
df2['num_clients'] = df[2]
df2['batch_size'] = df[3]
df2['sgx'] = df[4]
df2['ssl'] = df[5]
df2['fl_time'] = df[len(df.columns)-1]
df2['train_test_time'] = sum
df2['comm'] = df[len(df.columns)-1]-sum

df2.set_index(['epochs','num_clients','batch_size','sgx','ssl'])
df3 = df2.groupby(['epochs','num_clients','batch_size','sgx','ssl'])['comm'].agg(['median','mean','max','min'])
df3['train_test_time'] = df2.groupby(['epochs','num_clients','batch_size','sgx','ssl'])['train_test_time'].mean()
df3['fl_time'] = df2.groupby(['epochs','num_clients','batch_size','sgx','ssl'])['fl_time'].mean()
if os.stat("client_result_agg.txt").st_size == 0:
    df3.to_csv("client_result_agg.txt")
else:
    df3.to_csv("client_result_agg.txt", mode='a', header=False)