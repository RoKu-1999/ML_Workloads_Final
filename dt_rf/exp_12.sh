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

export OMP_NUM_THREADS="1"
echo $OMP_NUM_THREADS

# Check for necessary commands
command -v numactl >/dev/null 2>&1 || { echo "numactl command not found. Exiting."; exit 1; }
command -v gramine-sgx >/dev/null 2>&1 || { echo "gramine-sgx command not found. Exiting."; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 command not found. Exiting."; exit 1; }

# Initialize the output file
# > results/result_dt.csv

echo 'model,sgx,max_depth,num_trees,train_percent,features,train_rows,train_time,test_rows,test_time,acc,f1' > results/result_rf.csv
train_size=70
# Loop through the train_sizes
for num_trees in 2 4 8 16 32 64 128; do
    numactl --physcpubind=0 --membind=0,1 -- python3 rf.py datasets_all/HIGGS.csv 1 5 0 0 $num_trees $train_size 1
    # Execute the first command
    numactl --physcpubind=0 --membind=0,1 -- gramine-sgx ./sklearnex ./rf.py datasets_all/HIGGS.csv 1 5 1 0 $num_trees $train_size 1
    # data.csv >> results/result_dt.csv
done

cp results/result_rf.csv ../final_results/rf/num_trees_bootstrap.csv

