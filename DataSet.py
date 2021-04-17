from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from skimage import io
import cv2

#https://blog.csdn.net/zhpf225/article/details/103766493


class MNIST_Data(Dataset):
    """
        读取数据、初始化数据
    """
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.train_txt = os.path.join(root_dir, 'rawtrain.txt')
        self.test_txt = os.path.join(root_dir, 'rawtest.txt')
        self.transform = transform
        if self.mode == 'train':
            with open(self.train_txt, "r") as f:
                self.data = f.readlines()
        if self.mode == 'test':
            with open(self.test_txt, "r") as f:
                self.data = f.readlines()
        # 去掉每一行的空格
        self.data = [line.strip("\n") for line in self.data]

    def __getitem__(self, index):
        img_path = self.data[index].split(' ')[0]
        target = int(self.data[index].split(' ')[-1])
        img = cv2.imread(img_path)
        img = img.transpose(2, 0, 1)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    pass
    # import matplotlib.pyplot as plt
    # mnist_data = MNIST_Data('./Data/MNIST/')
    # train_loader = DataLoader(dataset=mnist_data,
    #                           batch_size=1,
    #                           shuffle=True)
    # data_iter = iter(train_loader)
    # img, label = data_iter.next()
    # plt.imshow(img.squeeze())
    # plt.show()
    # print(img.shape)
    # print(label)
