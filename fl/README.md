# Federated Learning Experiments

To run experiments for FL Training:

    ./env.sh

Inside tmux run:
 
    run_exp -n 0,1 -t 01:30:00 -m "fl workload" .expX/exp_X.sh


In order to run Experiment 21 with Asynchronous Enclave Call Feature, pls uncomment following line and run experiment 20:

    sgx.insecure__rpc_thread_num = 4
