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

echo 'model,sgx,max_depth,MiB,num_features,num_rows,train_time,cpu_time' > cpu-times.txt
# Initialize the output file

make clean
make SGX=1 DEBUG=1

max_depth=3
dataset=datasets/HIGGS.csv
# Loop through the train_sizes
for train_size in 10 90; do
    # Execute the first command
    # Execute the second command
    > results/result_dt.csv

    cpu_time2=$((time numactl --physcpubind=0 --membind=0,1 -- python3 dt.py datasets_all/HIGGS.csv 1 0 $max_depth 70) 2>&1 | grep user | awk '{split($2, a, "m"); split(a[2], b, "s"); print a[1]*60 + b[1]}')
    if [[ -f results/result_dt.csv && -n "$cpu_time2" ]]; then
        while IFS= read -r line; do
            echo "$line,$cpu_time2"
        done < results/result_dt.csv >> cpu-times.txt
    else
        echo "Error during the execution of the second command or data.csv not found for train_size $train_size"
    fi

    > results/result_rf.csv

    cpu_time1=$((time numactl --physcpubind=0 --membind=0,1 -- gramine-sgx ./sklearnex ./dt.py datasets_all/HIGGS.csv 1 1 $max_depth 70) 2>&1 | grep user | awk '{split($2, a, "m"); split(a[2], b, "s"); print a[1]*60 + b[1]}')
    if [[ -f results/result_dt.csv && -n "$cpu_time1" ]]; then
        while IFS= read -r line; do
            echo "$line,$cpu_time1"
        done < results/result_dt.csv >> cpu-times.txt
    else
        echo "Error during the execution of the first command or data.csv not found for train_size $train_size"
    fi
done