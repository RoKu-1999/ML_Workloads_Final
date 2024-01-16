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
iterations=11

make clean
make SGX=1 DEBUG=1

echo 'model,sgx,columns,train_rows,train_time,test_rows,test_time,acc,f1' > results/result_nb.csv
python3 nb.py datasets_all/HIGGS.csv $iterations 0
gramine-sgx ./sklearnex ./nb.py datasets_all/HIGGS.csv $iterations 1
python3 nb.py datasets_all/Android_Malware.csv $iterations 0
gramine-sgx ./sklearnex ./nb.py datasets_all/Android_Malware.csv $iterations 1
python3 nb.py datasets_all/card_transdata.csv $iterations 0
gramine-sgx ./sklearnex ./nb.py datasets_all/card_transdata.csv $iterations 1

cp results/result_nb.csv ../final_results/nb/result_nb.csv
