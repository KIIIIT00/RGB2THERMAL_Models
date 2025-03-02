import cv2

dataset_path = './CSTGAN/datasets/Scene2ver2/trainB/thermal_1.jpg'

image = cv2.imread(dataset_path)
print(image.shape)