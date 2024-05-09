#!/bin/bash
ls  /usr/lib/x86_64-linux-gnu/libcuda.so.1 -la
dd_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
echo "version:$dd_version"
cd /usr/lib/x86_64-linux-gnu
ln -f -s libcuda.so.$dd_version libcuda.so.1

