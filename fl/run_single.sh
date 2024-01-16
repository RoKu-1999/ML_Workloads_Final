#!/bin/bash
# Download the CIFAR-10 dataset
python -c "from torchvision.datasets import CIFAR10; CIFAR10('./data', download=True)"

num_cores=$(($(nproc --all)-1))
# Params: 
# $1 = Number of Epochs
# $2 = Batch Size Parameter
# $3 = Number of Clients
# $4 = SGX / Native Execution
# $5 = SSL Enabled
epochs=$1
batch_size=$2
num_clients=$3
ssl=$5
echo "--------------------------------- Starting Workload $4 ---------------------------------"

echo "$(date +%s): Epochs: $epochs, batch: $batch_size, num_clients: $num_clients"
sgx=1
if [ "$4" == "sgx" ]; then
    numactl --physcpubind=0-7 --membind=0 -- gramine-sgx ./pytorch server.py $batch_size $num_clients $ssl $epochs $sgx &
    sleep 300
else
    sgx=0
    numactl --physcpubind=0-7 --membind=0 -- python3 server.py $batch_size $num_clients $ssl $epochs $sgx &
    sleep 3
fi 

echo "started Server"
offset=-1
for i in $(seq 1 $num_clients); do
    j=$((i + 7))
    echo "start client ($j)"
    numactl --physcpubind=$j --membind=0,1 -- python3 client.py $batch_size $num_clients $i $ssl $epochs $sgx &
    pid=$!
done

wait $pid
sleep 10

#offset=0
#for pid in ${pids[*]}; do
#    if [ "$offset" == "$((num_clients-1))" ]; then
#        wait $pid
#    else
#        wait $pid &
#    fi
#    offset=$((offset+1))
#done
echo "--------------------------------- Done Workload ---------------------------------"