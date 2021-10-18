#train.py

import torch.nn as nn

import torch.optim as optim
from model import vgg
import torch
import numpy as np
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model_name = "vgg16"

net = vgg(model_name=model_name, num_classes=136, init_weights=True)
net.to(device)
loss_function = nn.MSELoss(size_average=None,
                 reduce=None,
                 reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=0.0001)


model_dir=r".\pretrain_models\face-keypoint-vgg16-0.pth"

checkpoint = torch.load(model_dir)
net.load_state_dict(checkpoint)
#
# predict use
net.eval()

images=cv2.imread(r".\test_img\test3.jpg")
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

