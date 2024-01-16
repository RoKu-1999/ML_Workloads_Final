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

make clean
make SGX=1 DEBUG=1

echo 'model,num_samples,sgx,gradient,num_threads,train_rows,train_time,test_rows,test_time,acc,f1' > results/result_svm.csv
for num_samples in 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000; do
    numactl --physcpubind=0-8 --membind=0 -- python3 svm.py $num_samples 0 11 0
    numactl --physcpubind=0-8 --membind=0 -- gramine-sgx ./sklearnex ./svm.py $num_samples 1 11 0
done

cp results/result_svm.csv ../final_results/svm/result_svm.csv
