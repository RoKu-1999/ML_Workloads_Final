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

echo 'model,sgx,max_depth,num_trees,train_percent,features,train_rows,train_time,test_rows,test_time,acc,f1' > results/result_rf.csv
file=datasets_all/HIGGS.csv
train_size=70

for max_depth in 3 5 7 9 11 13 15 20 30 40 50 0; do
    # Execute the first command
    numactl --physcpubind=0 --membind=0,1 -- gramine-sgx ./sklearnex ./rf.py datasets_all/HIGGS.csv 5 1 $max_depth 1 1 $train_size 1
    # data.csv >> results/result_dt.csv

    # Execute the second command
    numactl --physcpubind=0 --membind=0,1 -- python3 rf.py datasets_all/HIGGS.csv 5 0 $max_depth 1 1 $train_size 1
    # data.csv >> results/result_dt.csv
done

cp results/result_rf.csv ../final_results/rf/max_depth_HIGGS.csv