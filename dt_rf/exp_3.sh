#!/usr/bin/env bash

#export GP=$HOME/gramine-$installation
echo $GP
#export PATH=$PATH:$GP/bin
echo $PATH
#export PYTHONPATH=$GP/lib/python3.10/site-packages
echo $PYTHONPATH
#export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$GP/lib/x86_64-linux-gnu/pkgconfig
echo $PKG_CONFIG_PATH

export OMP_NUM_THREADS="1"
echo $OMP_NUM_THREADS

iterations=11

make clean
make SGX=1 DEBUG=1

echo 'model,sgx,max_depth,dataset,train_percent,features,train_rows,train_time,test_rows,test_time,acc,f1' > results/result_dt.csv
max_depth=3

for train_percent in 10 20 30 40 50 60 70 80 90; do
    # Your commands go here
    # For example, to print the name of each file
    python3 dt.py datasets_all/HIGGS.csv $iterations 0 $max_depth $train_percent
    gramine-sgx ./sklearnex ./dt.py datasets_all/HIGGS.csv $iterations 0 $max_depth $train_percent
done

cp results/result_dt.csv ../final_results/dt/train_test_increase.csv