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
file=datasets_all/HIGGS.csv

for max_depth in 3 5 7 9 11 13 15 0; do
    echo $file
    # Your commands go here
    # For example, to print the name of each file
    python3 dt.py $file $iterations 0 $max_depth 70
    gramine-sgx ./sklearnex ./dt.py $file $iterations 1 $max_depth 70
done

cp results/result_dt.csv ../final_results/dt/max_depth_HIGGS.csv