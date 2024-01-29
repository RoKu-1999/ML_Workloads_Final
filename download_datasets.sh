#!/bin/bash

# Define the URLs of the datasets
url1="https://www.kaggle.com/datasets/subhajournal/android-malware-detection"
url2="https://archive.ics.uci.edu/ml/datasets/HIGGS"
url3="https://www.kaggle.com/mlg-ulb/creditcardfraud"
url4="https://www.cs.toronto.edu/~kriz/cifar.html"

# Create a directory for the datasets
mkdir -p datasets_all

# Download the datasets using wget
wget -P datasets_all "$url1"
wget -P datasets_all "$url2"
wget -P datasets_all "$url3"
wget -P datasets_all "$url4"

# Find subfolders starting with 'exp_' and copy datasets into them
find . -type d -name 'exp_*' -exec cp -R datasets_all/* {} \;
