#train.py

import torch.nn as nn
from torchvision import transforms, datasets
import json
import os
import torch.optim as optim
from model import vgg
import torch
import time
import numpy as np
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# train_dataset = dataloader.MyDataset()
# Batch_size = 1
# train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=0, pin_memory=True,
#                  drop_last=True, collate_fn=dataloader.my_dataset_collate)

# #数据预处理，从头
# data_transform = {
#     "train": transforms.Compose([transforms.RandomResizedCrop(224),
#                                  transforms.RandomHorizontalFlip(),
#                                  transforms.ToTensor(),
#                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
#     "val": transforms.Compose([transforms.Resize((224, 224)),
#                                transforms.ToTensor(),
#                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
#
#
# #data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
# data_root = os.getcwd()
# image_path = data_root + "/101_ObjectCategories/"
# val_path = data_root + "/101_ObjectCategories_val/"
#
#
# batch_size = 1
# train_dataset = datasets.ImageFolder(root=image_path,
#                                      transform=data_transform["train"])
# train_num = len(train_dataset)
# train_loader = torch.utils.data.DataLoader(train_dataset,
#                                            batch_size=batch_size, shuffle=True,
#                                            num_workers=0)
#
# validate_dataset = datasets.ImageFolder(root=val_path,
#                                         transform=data_transform["val"])
# val_num = len(validate_dataset)
# validate_loader = torch.utils.data.DataLoader(validate_dataset,
#                                               batch_size=batch_size, shuffle=False,
#                                               num_workers=0)
#
#
# # test_data_iter = iter(validate_loader)
# # test_image, test_label = test_data_iter.next()
#
model_name = "vgg16"

net = vgg(model_name=model_name, num_classes=136, init_weights=True)
net.to(device)
loss_function = nn.MSELoss(size_average=None,
                 reduce=None,
                 reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=0.0001)
#
# best_acc = 0.0
# save_path = './{}Net.pth'.format(model_name)
#
test_flag = True
model_dir=r".\best_models\vgg16Net-0.pth"
if test_flag:
    # 加载保存的模型直接进行测试机验证，不进行此模块以后的步骤
    checkpoint = torch.load(model_dir)
    net.load_state_dict(checkpoint)
#
# predict use
net.eval()

images=cv2.imread(r".\test_img\test4.jpg")
images = cv2.resize(images, (224, 224))
images_input = torch.from_numpy(images)
images_input =  images_input.unsqueeze(0)
images_input=np.transpose(images_input,(0,3,1,2))
images_input = images_input.float()
outputs = net(images_input.to(device))

outputs_val = torch.squeeze(outputs)
print(outputs_val)
for p in range(68):
    cv2.circle(images, (int(outputs_val[p * 2] * 224), int(outputs_val[p * 2 + 1] * 224)),
               2, (0, 255, 0), 2)

cv2.imshow("11", images)
cv2.waitKey(0)

