#!/bin/bash
#Benchmark
git lfs install
git clone https://huggingface.co/datasets/nyu-visionx/CV-Bench
#Reference
wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
unzip CLEVR_v1.0.zip "CLEVR_v1.0/images/val/*" -d clevr
unzip reference.zip
