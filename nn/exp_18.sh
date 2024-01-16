#!/usr/bin/env bash

set -e

#make clean
#make SGX=1 DEBUG=1
# Delete old log file and create new one
rm logs.txt
touch logs.txt


# Export environment variables : if you want to build another installation, edit installation
# installation=bin
# export GP=$HOME/gramine-$installation
echo $GP
#export PATH=$PATH:$GP/bin
echo $PATH
#export PYTHONPATH=$GP/lib/python3.10/site-packages
echo $PYTHONPATH
#export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$GP/lib/x86_64-linux-gnu/pkgconfig
echo $PKG_CONFIG_PATH

echo 'sgx,arch,batch_size,batch_eval' > result_batch_eval.txt
echo 'sgx,arch,batch_size,to_dev,output,acc,sgd_step,one_batch' > result_batch_train.txt
echo 'sgx,threads,arch,batch_size,epoch_eval' > result_epoch_eval.txt
echo 'sgx,threads,arch,batch_size,epoch_train' > result_epoch_train.txt

make clean
make SGX=1 DEBUG=1

#numactl --cpunodebind=0 --membind=0 -- gramine-sgx ./pytorch ./main.py 1 -a vgg19 --dummy --epochs 1 --batch-size $batch_size -j 0 --seed 42
for batch_size in 16 64 128; do
    for threads in 1 2 4 8 16 32 64; do
        export OMP_NUM_THREADS=$threads
        if [ "$threads" -eq 1 ]; then
            cpu_bind=0
        else
            # Generate a string that lists the cores from 0 to threads-1, separated by commas.
            cpu_bind=$(seq -s ',' 0 $((threads - 1)))
        fi
        echo "cpu bind : $cpu_bind"
        
        numactl --physcpubind=$cpu_bind --membind=0 -- gramine-sgx ./pytorch ./main.py 1 $threads -a resnet18 --dummy --epochs 5 --batch-size $batch_size -j 0 --seed 42
        numactl --physcpubind=$cpu_bind --membind=0 -- gramine-sgx ./pytorch ./main.py 1 $threads -a resnet34 --dummy --epochs 5 --batch-size $batch_size -j 0 --seed 42
        numactl --physcpubind=$cpu_bind --membind=0 -- gramine-sgx ./pytorch ./main.py 1 $threads -a resnet50 --dummy --epochs 5 --batch-size $batch_size -j 0 --seed 42
        numactl --physcpubind=$cpu_bind --membind=0 -- gramine-sgx ./pytorch ./main.py 1 $threads -a alexnet --lr 0.01 --dummy --epochs 5 --batch-size $batch_size -j 0 --seed 42
        numactl --physcpubind=$cpu_bind --membind=0 -- gramine-sgx ./pytorch ./main.py 1 $threads -a vgg19 --dummy --epochs 5 --batch-size $batch_size -j 0 --seed 42
 
        numactl --physcpubind=$cpu_bind --membind=0 -- python3 main.py 0 $threads -a resnet18 --dummy --epochs 5 --batch-size $batch_size -j 0 --seed 42
        numactl --physcpubind=$cpu_bind --membind=0 -- python3 main.py 0 $threads -a resnet34 --dummy --epochs 5 --batch-size $batch_size -j 0 --seed 42
        numactl --physcpubind=$cpu_bind --membind=0 -- python3 main.py 0 $threads -a resnet50 --dummy --epochs 5 --batch-size $batch_size -j 0 --seed 42
        numactl --physcpubind=$cpu_bind --membind=0 -- python3 main.py 0 $threads -a alexnet --lr 0.01 --dummy --epochs 5 --batch-size $batch_size -j 0 --seed 42
        numactl --physcpubind=$cpu_bind --membind=0 -- python3 main.py 0 $threads -a vgg19 --dummy --epochs 5 --batch-size $batch_size -j 0 --seed 42
   done
done

cp result_epoch_eval.txt ../final_results/nn/result_epoch_eval.txt
cp result_epoch_eval.txt ../final_results/nn/result_epoch_train.txt

echo "Finished workload: Results can be found under folder final_results/nn/"