#!/bin/bash

# 训练模型 a
python gan/GANs/GC_network_chz_dataloader.py --bottleneck 4 --epoch 1026 --savedir '/chz/res/ChzKodakGC/train'
# 训练模型 b
sleep 600
python gan/GANs/GC_network_chz_dataloader.py --bottleneck 8 --epoch 1026 --savedir '/chz/res/ChzKodakGC/train'
sleep 600
python gan/GANs/GC_network_chz_dataloader.py --bottleneck 16 --epoch 1026 --savedir '/chz/res/ChzKodakGC/train'
sleep 600
python gan/GANs/GC_network_chz_dataloader.py --bottleneck 32 --epoch 1026 --savedir '/chz/res/ChzKodakGC/train'
sleep 600
