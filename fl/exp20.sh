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

num_epoch=10
batch_size=32
ssl=1

for z in `seq 0 2`; do
    for num_clients in 2 8 16 24 25 26 32 40 48 55; do
        ./run_single.sh $num_epoch $batch_size $num_clients "sgx" $ssl
        wait $!
    done
done

for z in `seq 0 2`; do
    for num_clients in 2 8 16 24 25 26 32 40 48 55; do
        ./run_single.sh $num_epoch $batch_size $num_clients "native" $ssl
        wait $!
    done
done