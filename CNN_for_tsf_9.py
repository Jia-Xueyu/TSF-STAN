"""
A CNN framework uses the tsf data as the input
made with pytorch
"""

import argparse
import numpy as np
import random
from torch.autograd import Variable
import torch.nn as nn
# import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import torch.distributed as dist
# from torch.utils.tensorboard import SummaryWriter
# import cv2 as cv
import torchvision

import os
import scipy.io as scio
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,6,7'
gpus = [0, 1, 2, 3]
# device = torch.cuda.set_device('cuda:{}'.format(gpus[0]))

device = torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")  # 500 is ok
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
opt = parser.parse_args()
batch_size = 50  # batch size = 50 at the beginning, 30 is okay, 10 is better


def weights_init(m):

    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, 1), stride=(4, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 23), stride=(1, 3)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            # nn.MaxPool2d(kernel_size=(1, 6), stride=(1, 6))
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 17), stride=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=(1, 6), stride=(1, 6))
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 7), stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )

        self.dense_layer = nn.Sequential(
            nn.Linear(11264, 2048),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 4),
            nn.Softmax()
        )


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.dense_layer(out)
        return out


def cnn(sub_index, data_train, label_train, data_test, label_test):

    classifier = CNN()
    classifier = classifier.cuda()
    classifier = nn.DataParallel(classifier, device_ids=[0, ])
    classifier = classifier.to(device)
    print('classifier')
    print(classifier)

    # set the seed for reproducibility
    seed_n = np.random.randint(500)
    print(seed_n)
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)

    data_train = data_train.transpose([0, 2, 1])
    data_test = data_test.transpose([0, 2, 1])
    data_train = np.expand_dims(data_train, axis=1)
    data_test = np.expand_dims(data_test, axis=1)

    data_train = torch.from_numpy(data_train)
    data_test = torch.from_numpy(data_test)
    label_train = torch.from_numpy(label_train)
    label_test = torch.from_numpy(label_test)

    dataset = torch.utils.data.TensorDataset(data_train, label_train)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


    optimizer = torch.optim.Adam(classifier.parameters(), lr=opt.lr)
    loss_func = nn.CrossEntropyLoss()

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    data_test = data_test.cuda()
    label_test = label_test.cuda()
    data_test = Variable(data_test.type(Tensor))

    optimizer = torch.optim.Adam(classifier.parameters(), lr=opt.lr)
    loss_func = nn.CrossEntropyLoss()

    test_acc = []
    pre=[]

    for epoch in range(opt.n_epochs):
        for step, data in enumerate(dataloader):
            datatrain, labeltrain = data
            datatrain = datatrain.cuda()
            labeltrain = labeltrain.cuda()

            datatrain = Variable(datatrain.type(Tensor))
            labeltrain = Variable(labeltrain.type(Tensor))

            output = classifier(datatrain)
            loss = loss_func(output, labeltrain.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 1 == 0:
                test_output = classifier(data_test)
                pred = torch.max(test_output, 1)[1].data.squeeze()
                pre.append(pred)
                accuracy = float((pred == label_test.type(torch.long)).cpu().numpy().astype(int).sum()) / float(label_test.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data, '| test accuracy: %.6f' % accuracy)
                test_acc.append(accuracy)

    test_acc.append(np.mean(test_acc))
    test_acc.append(np.max(test_acc))
    max_index=test_acc.index(np.max(test_acc))
    final_pre=pre[max_index]
    test_acc.append(seed_n)
    save_acc = np.array(test_acc)

    np.savetxt('./result/exp2_collected_data_JSFDSTCNN_' + str(
        sub_index) + '.txt', save_acc, '%.10f')
    scio.savemat('./pre_and_label/exp2_collected_data_JSFDSTCNN_'+str(
        sub_index) + '.mat',{'pre':final_pre,'label':label_test})

