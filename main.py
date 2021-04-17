import torchvision
import torchvision.transforms as transforms
import torchvision.datasets.mnist as mnist
from torch.utils.data import DataLoader
import os
from skimage import io
"""
下载MNIST数据并转化为jpg图像
"""


batch_size = 16

# Data set
train_dataset = torchvision.datasets.MNIST(root='./Data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=False)

test_dataset = torchvision.datasets.MNIST(root='./Data',
                                          train=False,
                                          transform=transforms.ToTensor(),
                                          download=False)

# Data loader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                        batch_size=batch_size,
                        shuffle=False)


def convert_to_img(root, train=True):
    if(train):
        f = open(root+'train.txt', 'w')
        data_path = root+'/train/'
        if(not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
            img_path=data_path+str(i)+'.jpg'
            io.imsave(img_path, img.numpy())
            f.write(img_path+' '+str(label.numpy())+'\n')
        f.close()
    else:
        f = open(root + 'test.txt', 'w')
        data_path = root + '/test/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img,label) in enumerate(zip(test_set[0],test_set[1])):
            img_path = data_path+ str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            f.write(img_path + ' ' + str(label.numpy()) + '\n')
        f.close()


if __name__=="__main__":
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    # print(images.shape)
    # print(labels.shape)

    root = "./Data/MNIST/raw"
    train_set = (
        mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
        mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
    )
    test_set = (
        mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
        mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
    )

    print("training set :", train_set[0].size())
    print("test set :", test_set[0].size())

    convert_to_img(root, True)  # 转换训练集
    convert_to_img(root, False)  # 转换测试集
