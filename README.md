# MLWorkloads_SGX
Master Thesis Robert
  
Template for scikit-learn is taken from: https://github.com/gramineproject/examples/tree/master/scikit-learn-intelex

### Datasets
  - https://www.kaggle.com/datasets/subhajournal/android-malware-detection
  - https://archive.ics.uci.edu/ml/datasets/HIGGS (2.6GB)
  - https://www.kaggle.com/mlg-ulb/creditcardfraud (150MB)

### Experiments

  - Experiments can be found in the subfolders
  - Must be run in tmux environment, which sets the environment variables (Start with ./env.sh

| Experiment | Model | Description                        |
|------------|-------|------------------------------------|
| 1          | NB    | Different Datasets                 |
| 2          | DT    | Different Datasets                 |
| 3          | DT    | Training Size Increase             |
| 4          | DT    | Relative Function Sampling         |
| 5          | DT    | Absolute Function Sampling         |
| 6          | DT    | Tree Depth Increase                |
| 7          | DT    | Relative Function Sampling         |
| 8          | DT/RF | Training Size Increase             |
| 9          | DT/RF | Event Counter Comparison           |
| 10         | RF    | Tree Depth Increase                |
| 11         | RF    | Number of Trees with Bootstrap     |
| 12         | RF    | Number of Trees without Bootstrap  |
| 13         | RF    | Function Sampling                  |
| 14         | RF    | Multithreading Bootstrap           |
| 15         | RF    | Multithreading without Bootstrap   |
| 16         | RF    | Cache Misses Single vs. 32 Threads |
| 17         | SVM   | Training and Inference             |
| 18         | NN    | Training and Inference             |
| 19         | FL    | Communication Epoch Increase       |
| 20         | FL    | Communication Client Increase      |
| 21         | FL    | Asynchronous Enclave Calls         |

### Building and installing gramine from source:
- uninstall gramine
- pull gramine project
- remove build dir
- run: meson setup build/ --buildtype=release --prefix=$HOME/gramine-bin -Ddirect=enabled -Dsgx=enabled
  - See meson flags for debug mode
- run: ninja -C build
- run: ninja -C build install
- env setup:
  - export GP=$HOME/gramine-bin
  - export PATH=$PATH:$GP/bin
  - export PYTHONPATH=$GP/lib/python3.8/site-packages (important: change version of python, i had python3.10)
  - export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$GP/lib/x86_64-linux-gnu/pkgconfig
- IMPORTANT: import pandas as pd need timezone information. Older versions of pytz and tzdata are deprecated, so if you get an error for timezonefiles missing
  - upgrade pytz up to the latest pytz-2022.2.1
  - pip install pytz --upgrade
  - upgarde tzdata up to tzdata-2022.2
  - pip install tzdata --upgrade

### Building and installing gramine as debug version:
  --buildtype=release

### Building and installing perf-tool from source:
  - uname -r => gives you your kernel version
  - git clone --single-branch --branch linux-5.4.y \
    https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git => replace 5.4 with uname -r (6.2 for me)
  - cd linux/tools/perf + make => you need to have flex
  - enable required features : for sure libdw-dev, libunwind-dev
  - install in DESTDIR: make install DESTDIR=$DESTDIR


### Using Perf tool:
  - First of all: It is important to add debug enclave (not log level, but enclave) to manifest because other manifest flags need an debug enclave (sgx.debug = true) => sgx.debug = false is a production enclave
  1. ## Enabling per-thread and process-wide SGX-stats
    - sgx.enable_stats = true in manifest
    - perf stat gramine-sgx helloworld in run_tests.sh
  2. ## SGX Profiling
    - Compile Gramine with -Dsgx=enabled --buildtype=debugoptimized
    - sgx.profile.enable = "main" or "all" for process dump
    - sgx.profile.with_stack = true for call chain information
    - Saves file to: sgx-perf.data
