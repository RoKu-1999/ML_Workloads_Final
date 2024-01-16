#!/usr/bin/env bash

set -e
export PYTHONDONTWRITEBYTECODE=1
export GP=$HOME/gramine-debug
export PATH=$PATH:$GP/bin
export PYTHONPATH=$GP/lib/python3.10/site-packages
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$GP/lib/x86_64-linux-gnu/pkgconfig
export PATH=$PATH:$HOME/perf-tool/bin/perf
tmux new -s release_exec
#tmux new -d -s release_exec
#tmux send-keys -t release_exec "./run_tests.sh ; exit" Enter
