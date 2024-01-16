#!/bin/bash

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

#!/bin/bash

# Number of iterations
n=11
# Path to the dataset
dataset="datasets/HIGGS.csv"
# Output directory
output_dir="perf_stats"
# Create the output directory if it doesn't exist
mkdir -p $output_dir

# Outer loop for iterations
for ((iteration=1; iteration<=n; iteration++)); do
    # Loop through the values and execute both commands with perf stat
    echo "Running for value: $val"

    # Using Gramine-SGX
    echo "Executing with Gramine-SGX..."
    numactl --physcpubind=0 --membind=0,1 -- ./../../perf_tool/bin/perf stat -e cycles,cache-references,cache-misses,branch-instructions,branch-misses,instructions,ref-cycles --output=$output_dir/rf_sgx_${iteration}.csv --field-separator=, gramine-sgx ./sklearnex ./rf.py datasets_all/HIGGS.csv 1 1 1 3 1 70 0

    # Using Gramine-SGX
    echo "Executing with Gramine-SGX..."
    numactl --physcpubind=0-32 --membind=0,1 -- ./../../perf_tool/bin/perf stat -e cycles,cache-references,cache-misses,branch-instructions,branch-misses,instructions,ref-cycles --output=$output_dir/dt_sgx_${iteration}.csv --field-separator=, gramine-sgx ./sklearnex ./dt.py datasets_all/HIGGS.csv 1 1 3 70

    echo "======================================"
done

echo "All runs completed."

