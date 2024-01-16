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

train_size=70
echo 'model,sgx,max_depth,num_trees,train_percent,features,train_rows,train_time,test_rows,test_time,acc,f1' > results/result_rf.csv

for num_threads in 64 32 16 8 4 2 1; do
    # Define and export OMP_NUM_THREADS
    export OMP_NUM_THREADS=$num_threads

    # Create a string of CPU cores to bind to, starting from 0 to num_threads - 1
    physcpubind=$(seq -s ',' 0 $((num_threads - 1)))

    # Execute the first command with the specified CPU core binding
    numactl --physcpubind=$physcpubind --membind=0,1 -- gramine-sgx ./sklearnex ./rf.py datasets_all/HIGGS.csv 5 1 3 128 $num_threads $train_size 0
    # If you want to output to a file, uncomment the next line
    # >> results/result_dt_gramine.csv

    # Execute the second command with the specified CPU core binding
    numactl --physcpubind=$physcpubind --membind=0,1 -- python3 rf.py datasets_all/HIGGS.csv 5 0 3 128 $num_threads $train_size 0
    # If you want to output to a file, uncomment the next line
    # >> results/result_dt_python.csv
done

cp results/result_rf.csv ../final_results/rf/multithreads_no_bootstrap.csv


