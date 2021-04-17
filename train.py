from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from skimage import io
import cv2
import DataSet
import torch
import config
from resnet import resnet18
import torch.optim as optim
import torch.nn as nn
from utils import AverageMeter, init_weights
from sklearn.metrics import precision_score, recall_score, accuracy_score
from tensorboardX import SummaryWriter
from xlwt import Workbook
import torchvision.models
import torchvision.utils as vutil
import matplotlib.pyplot as plt


def evaluating(val_loader, net, criterion):
    val_loss_meter = AverageMeter()
    traget_list = []
    pred_list = []

    net.eval()
    with torch.no_grad():
        print("测试集数据量：", len(val_loader))
        for j, data in enumerate(val_loader):
            val_image, val_label = data
            input_val = val_image.float().cuda()
            target_val = val_label.long().cuda()  # (batch_size, H,W)
            val_output = net(input_val)

            val_loss = criterion(val_output, target_val)   # 损失函数有softmax
            val_loss_meter.update(val_loss.cpu().item())
            pred_val = torch.argmax(torch.softmax(val_output, 1), 1).long()  #

            traget_list.append(target_val.cpu().item())
            pred_list.append(pred_val.cpu().item())
        return val_loss_meter.average(), traget_list, pred_list


def main(is_train=True, is_infer=True):
    train_set = DataSet.MNIST_Data(my_args.root_dir, 'train')
    test_set = DataSet.MNIST_Data(my_args.root_dir, 'test')
    train_loader = DataLoader(train_set,
                              batch_size=my_args.batch_size,
                              num_workers=0,
                              shuffle=True,
                              pin_memory=True)
    test_loader = DataLoader(test_set,
                             batch_size=my_args.batch_size_val,
                             num_workers=0,
                             shuffle=False,
                             pin_memory=True)
    net = resnet18(pretrained=True)
    # init_weights(net)

    # ==================冻结参数训练==============================
    # name_list = ['layer1']  # list中为需要冻结的网络层
    # for name, value in net.named_parameters():  # 这里名称是卷积层细化的名称
    #     if name in name_list:
    #         value.requires_grad = False

    ct = 0
    for child in net.children():
        ct += 1
        if ct == 5:  # 只固定了layer1的参数进行训练
            # print(child)
            for param in child.parameters():
                param.requires_grad = False

    # 检查参数是否被固定
    # for k, v in net.named_parameters():
    #     print(k, v.requires_grad)  # 理想状态下，所有值都是False
    #######################################
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        net.cuda()
        criterion.cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=1e-5)
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=my_args.lr, amsgrad=True)

    if is_train:
        train_loss_meter = AverageMeter()
        for epoch in range(0, my_args.epochs):
            print('================epoch{}================='.format(epoch))
            # switch to train mode
            net.train()
            for i, sample_batched in enumerate(train_loader):
                iter = epoch * len(train_loader) + i
                image, label = sample_batched
                input_var = image.float()  # B,3,H,W
                input_var = input_var.cuda()
                target_var = label.long()
                target_var = target_var.cuda()  # B,H,W

                output = net(input_var)  # 输出没有激活
                train_loss = criterion(output, target_var)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()  # 更新网络参数
                # print('train_loss_{}'.format(train_loss))
                train_loss_meter.update(train_loss.cpu().item())
            # train_writer.add_scalar('loss', train_loss_meter.average(), epoch)

            # 在测试集上测试
            test_loss, traget_list, pred_list = evaluating(test_loader, net, criterion)
            # 计算指标：
            acc = accuracy_score(traget_list, pred_list)
            # precision = precision_score(traget_list, pred_list)
            # recall = recall_score(traget_list, pred_list)
            # print("acc: ", acc, " precision:", precision, " recall:", recall)

            # test_writer.add_scalar('loss', test_loss, epoch)
            # test_writer.add_scalar('acc', acc, epoch)
            # test_writer.add_scalar('precision', precision, epoch)
            # test_writer.add_scalar('recall', recall, epoch)
            print("train_loss:", train_loss_meter.average(),
                  "test_loss", test_loss,
                  "acc", acc)

        # 保留最后一个模型
        torch.save(net.state_dict(), os.path.join(model_path, 'Final_model' + my_args.modelName + ".pth.tar"))
        torch.save(net, os.path.join(model_path, 'All_Final_model' + my_args.modelName + ".pth.tar"))

    if is_infer:
        # 在推理阶段实现特征图的保存
        trained_model_path = os.path.join(model_path, 'Final_model' + my_args.modelName + '.pth.tar')
        net.load_state_dict(torch.load(trained_model_path))
        _, traget_list, pred_list = evaluating(test_loader, net, criterion)
        # 计算指标
        acc = accuracy_score(traget_list, pred_list)
        print("acc:", acc)
        # 将结果写为xls文件
        write_excel(traget_list, pred_list, xls_path)
        # 提取中间层特征图
        feature_output1 = net.feature_map.transpose(1, 0).cpu()
        out = torchvision.utils.make_grid(feature_output1)
        feature_imshow(out)


def feature_imshow(inp, title=None):
    """Imshow for Tensor."""

    inp = inp.detach().numpy().transpose((1, 2, 0))
    # mean = np.array([0.5, 0.5, 0.5])
    #
    # std = np.array([0.5, 0.5, 0.5])
    #
    # inp = std * inp + mean
    #
    # inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def make_excel(name):
    if not os.path.exists(name):
        book = Workbook(encoding='utf-8')
        book.add_sheet('Sheet 1')
        book.save(name)


def write_excel(target_list, pred_list, file_path):
    f = open(file_path, 'w')
    for i, case in enumerate(zip(target_list, pred_list)):
        f.write(str(case) + "\n")

    f.close()


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def hook_func(module, input, output):
    """
    Hook function of register_forward_hook
    Parameters:
    -----------
    module: module of neural network
    input: input of module
    output: output of module
    """
    image_name = get_image_name_for_hook(module)
    data = output.clone().detach()
    data = data.permute(1, 0, 2, 3)
    vutil.save_image(data, image_name, pad_value=0.5)


def get_image_name_for_hook(module):
    """
    Generate image filename for hook function
    Parameters:
    -----------
    module: module of neural network
    """
    os.makedirs(INSTANCE_FOLDER, exist_ok=True)
    base_name = str(module).split('(')[0]
    index = 0
    image_name = '.'  # '.' is surely exist, to make first loop condition True
    while os.path.exists(image_name):
        index += 1
        image_name = os.path.join(
            INSTANCE_FOLDER, '%s_%d.png' % (base_name, index))
    return image_name


if __name__ == '__main__':
    INSTANCE_FOLDER = None
    np.random.seed(1)
    torch.manual_seed(1)  # 同时设置CPU和GPU的随机种子
    torch.cuda.empty_cache()
    my_args = config.get_config()
    log_path = os.path.join(my_args.log_dir, my_args.modelName)
    make_dir(log_path)
    train_log_path = os.path.join(log_path, 'train')
    make_dir(train_log_path)
    test_log_path = os.path.join(log_path, 'test')
    make_dir(test_log_path)
    # train_writer = SummaryWriter(train_log_path)
    # test_writer = SummaryWriter(test_log_path)

    model_path = os.path.join(my_args.model_dir, my_args.modelName)
    make_dir(model_path)
    xls_path = os.path.join(my_args.root_dir+'result.xls')
    print(xls_path)
    make_excel(xls_path)

    main(is_train=False, is_infer=True)
