# coding: utf-8

import argparse


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelName", default="ResNet18", type=str)   # 这里没有改
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--batch_size_val', default=1, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--weight_decay', default=0.00001, type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--model_dir', default='./Model', type=str)
    parser.add_argument('--root_dir', default='./Data/MNIST/', type=str)
    parser.add_argument('--log_dir', default='./Log', type=str)

    args = parser.parse_args()
    return args
