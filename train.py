#train.py

import torch.nn as nn
from torchvision import transforms, datasets
import json
import os
import torch.optim as optim
from model import vgg
import torch
import time
import data_loader
import numpy as np
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


train_dataset = data_loader.MyDataset()
Batch_size = 4
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=0, pin_memory=True,
                 drop_last=True, collate_fn=data_loader.my_dataset_collate)


model_name = "vgg16"

net = vgg(model_name=model_name, num_classes=136, init_weights=True)
net.to(device)
loss_function = nn.MSELoss(size_average=None,
                 reduce=None,
                 reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=0.0001)

test_flag = True
model_dir=r".\best_models\vgg16Net-0.pth"
if test_flag:
    # 加载保存的模型直接进行测试机验证，不进行此模块以后的步骤
    checkpoint = torch.load(model_dir)
    net.load_state_dict(checkpoint)
#

print("start to train")
for epoch in range(100):
    # train
    net.train()
    running_loss = 0.0
    t1 = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        labels = np.array(labels)
        optimizer.zero_grad()

        labels_val = labels[0]
        images_val = np.squeeze(images)
        # for p in range(68):
        #     cv2.circle(images_val, (int(labels_val[p * 2] * 224), int(labels_val[p * 2 + 1] * 224)),
        #                2, (0, 255, 0), 2)
        #
        # cv2.imshow("11", images_val)
        # cv2.waitKey(0)
        #with torch.no_grad(): #用来消除验证阶段的loss，由于梯度在验证阶段不能传回，造成梯度的累计

        images = torch.from_numpy(images)

        labels = torch.from_numpy(labels)
        images=np.transpose(images,(0,3,1,2))

        #print(images.shape)
        images=images.float()
        labels = labels.float()
        outputs = net(images.to(device))

        outputs_val = torch.squeeze(outputs)
        #print(outputs_val)
        # print(outputs_val)
        # for p in range(68):
        #     cv2.circle(images_val, (int(outputs_val[p * 2] * 224), int(outputs_val[p * 2 + 1] * 224)),
        #                2, (0, 255, 0), 2)
        #
        # cv2.imshow("11", images_val)
        # cv2.waitKey(0)

        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        print('epoch : {} step : {} batch_loss: {:.8f}, learning_rate: {}'.format(epoch,step,loss.data,optimizer.state_dict()['param_groups'][0]['lr']))

    save_path = './{}Net-{}.pth'.format(model_name,epoch)
    if epoch % 1 == 0:
        torch.save(net.state_dict(), save_path)

print('Finished Training')
