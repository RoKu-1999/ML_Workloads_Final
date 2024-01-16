#!/usr/bin/env bash

rm logs.txt
touch logs.txt
chmod +rwx logs.txt
#export GP=$HOME/gramine-$installation
echo $GP
#export PATH=$PATH:$GP/bin
echo $PATH
#export PYTHONPATH=$GP/lib/python3.10/site-packages
echo $PYTHONPATH
#export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$GP/lib/x86_64-linux-gnu/pkgconfig
echo $PKG_CONFIG_PATH

#export OMP_NUM_THREADS="1"
#echo $OMP_NUM_THREADS
iterations=1

dataset=datasets/HIGGS.csv
max_depth=3
echo 'percent_spent,function,sgx,train_size' > top_10_functions.csv

make clean
make SGX=1 DEBUG=1

for train_size in 10 20 30 40 50 60 70 80 90; do
    # First command set with sgx=1
    sgx=1
    numactl --physcpubind=0 --membind=0,1 -- gramine-sgx ./sklearnex ./dt.py $dataset $iterations 1 $max_depth $train_size
    ./../../perf_tool/bin/perf report -i sgx-perf.data | awk -v sgx="$sgx" -v ts="$train_size" 'NR>11 && NR<=26 {gsub("%", "", $1); print $1/100 "," $5 "," sgx "," ts}' >> top_10_functions.csv
    rm sgx-perf.data
    #mv sgx-perf.data sgx-perf_$trees.data
    # Second command set with sgx=0
    sgx=0
    numactl --physcpubind=0 --membind=0,1 -- ./../../perf_tool/bin/perf record -e cpu-clock:HGu -o perf.data python3 dt.py $dataset $iterations 0 $max_depth $train_size
    ./../../perf_tool/bin/perf report -i perf.data | awk -v sgx="$sgx" -v ts="$train_size" 'NR>11 && NR<=26 {gsub("%", "", $1); print $1/100 "," $5 "," sgx "," ts}' >> top_10_functions.csv
    rm perf.data
    #mv perf.data perf_$trees.data
done