#!/bin/bash

# This experiment runs with following settings:
#
# Number of epochs is constantly 5
# Batch Size is constantly 32
# Number of clients grows from 2 to 127
# SSL Connection is enabled
# 
# Params for single execution: 
# $1 = Number of Epochs
# $2 = Batch Size Parameter
# $3 = Number of Clients
# $4 = SGX / Native Execution
# $5 = SSL Enabled
#
# ./../run_single.sh 3 32 2 "native" 1

batch_size=32
ssl=1
num_clients=16

for z in `seq 0 11`; do
    for num_epoch in 1 5 10 15 20 15; do
        ./run_single.sh $num_epoch $batch_size $num_clients "sgx" $ssl
        wait $!
    done
done

for z in `seq 0 11`; do
    for num_epoch in 1 5 10 15 20 15; do
        ./run_single.sh $num_epoch $batch_size $num_clients "native" $ssl
        wait $!
    done
done