# face-key-point-pytorch

[![Python](https://img.shields.io/static/v1?label=build&message=passing&color=green)](https://www.python.org/)
[![Python](https://img.shields.io/static/v1?label=python&message=3.8.12&color=blue)](https://www.python.org/)
[![torch](https://img.shields.io/static/v1?label=torch&message=1.8.1&color=blue)](https://pytorch.org/)


## 1. Data structure

The structure of landmarks_jpg is like below:

```
|--landmarks_jpg
|----AFW
|------AFW_134212_1_0.jpg
|------AFW_134212_1_1.jpg
|----HELEN
|-------HELEN_232194_1_0.jpg
|-------HELEN_232194_1_1.jpg
|----IBUG
|------IBUG_image_003_1_0.jpg
|------IBUG_image_003_1_1.jpg
|----LFPW
|------LFPW_image_test_0001_0.jpg
|------LFPW_image_test_0001_1.jpg
```

The structure of landmarks_label is like below:

```
|--landmarks_label
|----AFW
|------AFW_134212_1_0_pts
|------AFW_134212_1_1_pts
|----HELEN
|-------HELEN_232194_1_0_pts
|-------HELEN_232194_1_1_pts
|----IBUG
|------IBUG_image_003_1_0_pts
|------IBUG_image_003_1_1_pts
|----LFPW
|------LFPW_image_test_0001_0_pts
|------LFPW_image_test_0001_1_pts
```

You can download it by yourself. You can also download the data from the cloud drive:

| name                | link |
| ------------------- | ---- |
| landmarks_jpg.zip   |  https://pan.baidu.com/s/1AJKpa0ac-6ZPWBASiMv87Q code: nujr |
| landmarks_label.zip | 链接：https://pan.baidu.com/s/1wBAZMFkNQS6R6KLkRl6ktw code: zgl0  |

## 2. how to train

Just simply run the below command:

```
python3 train.py
```



## 3. how to test 
Revise the test file name in predict.py and then run the below command:
```
python3 predict.py
```


