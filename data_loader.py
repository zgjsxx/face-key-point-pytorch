import cv2
import numpy as np
from PIL import Image
import glob
from scipy.io import loadmat
from torch.utils.data.dataset import Dataset

import torchvision.transforms as transforms
landmarks_anno_path = r"D:\python_proj\face-key-point-pytorch\landmarks_label"
landmarks_jpg_path = r"D:\python_proj\face-key-point-pytorch\landmarks_jpg"


class MyDataset(Dataset):
    def __init__(self, shape=[224, 224], is_train=True):
        self.shape = shape
        self.is_train = is_train
        landmark_path_folder = glob.glob(landmarks_anno_path + "\\*")
        #print(landmark_path_folder)
        landmark_anno_list = []
        for f in landmark_path_folder:
            landmark_anno_list += glob.glob(f + "\\*.mat")
        self.landmark_anno_list=landmark_anno_list

    def __len__(self):
        return self.landmark_anno_list.__len__()

    def get_random_data(self, index, random=True):
        landmark_info = self.landmark_anno_list[index]
        im_path = landmark_info.replace("landmarks_label", "landmarks_jpg").replace("_pts.mat", ".jpg")

        im_data = cv2.imread(im_path)
        m = loadmat(landmark_info)
        landmark = None
        if m.__contains__("pt2d"):
            landmark = m['pt2d']
        elif m.__contains__("pts_2d"):
            landmark = m['pts_2d']

        x_max = int(np.max(landmark[0:68, 0]))
        x_min = int(np.min(landmark[0:68, 0]))
        y_max = int(np.max(landmark[0:68, 1]))
        y_min = int(np.min(landmark[0:68, 1]))

        #
        y_min = int(y_min - (y_max - y_min) * 0.3)
        y_max = int(y_max + (y_max - y_min) * 0.05)
        x_min = int(x_min - (x_max - x_min) * 0.05)
        x_max = int(x_max + (x_max - x_min) * 0.05)

        # cv2.rectangle(im_data,(x_min,y_min),(x_max,y_max),(0,255,255),2)
        # cv2.imshow("11",im_data)
        # cv2.waitKey(0)

        face_data = im_data[y_min:y_max, x_min:x_max]

        sp = face_data.shape

        im_point = []

        for p in range(68):
            im_point.append((landmark[p][0] - x_min) * 1.0 / sp[1])
            im_point.append((landmark[p][1] - y_min) * 1.0 / sp[0])

            # cv2.circle(face_data, (int(im_point[p * 2] * sp[1]), int(im_point[p * 2 + 1] * sp[0])),
            #           2, (0, 255, 0), 2)

        # cv2.imshow("11", face_data)
        # cv2.waitKey(0)
        face_data = cv2.resize(face_data, (224, 224))
        return face_data,im_point


    def __getitem__(self, index):
        face_data,label = self.get_random_data(index)
        return face_data,label

def my_dataset_collate(batch):
    images = []
    labels = []
    for img,  label in batch:
        images.append(img)
        labels.append(label)
    images = np.array(images)
    return images, labels



if __name__=="__main__":
    dataset = MyDataset()
    face_data,label = dataset.get_random_data(index=1)
    print(face_data)
    print(label)

    # for p in range(68):
    #     cv2.circle(face_data, (int(label[p*2]*128), int(label[p * 2 + 1]*128)),
    #                2, (0, 255, 0), 2)
    # cv2.imshow("test",face_data)
    # cv2.waitKey(0)