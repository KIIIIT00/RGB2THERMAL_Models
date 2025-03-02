import random
import os
import numpy as np
import torch.utils.data as data
from PIL import Image
from options.base_options import BaseOptions
from data import CustomDatasetDataLoader
from data.single_dataset import SingleDataset
import torchvision.transforms as transforms, datasets
from torchvision.io import read_image

class ThermalImageResize:
    def __init__(self, input_dir, outout_dir):
        self.input_dir = input_dir
        self.output_dir = outout_dir
    
    def make_dir(self):
        if os.path.exists(self.output_dir):
            print(f"Warning: The output directory {self.output_dir} already exists.")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def resize_image(self, image):
        resize_transform = transforms.Resize((256, 256), transforms.InterpolationMode.BICUBIC)
        resized_image = resize_transform(image)
        return resized_image
    
    def resize_images_in_folder(self):
        self.make_dir()
        
        for filename in os.listdir(self.input_dir):
            input_image_path = os.path.join(self.input_dir, filename)
            output_image_path = os.path.join(self.output_dir, filename)
            
            image = Image.open(input_image_path)
            resized_image = self.resize_image(image)
            resized_image.save(output_image_path)
            print(f"saved image:{output_image_path}")
            
if __name__ == '__main__':
    THERMAL_IMAGE_DIR = './datasets/Scene2ver2/testB/'
    OUTPUT_THERMAL_DIR = './datasets/Scene2ver2/256x256_testB/'
    
    thermal_resize = ThermalImageResize(THERMAL_IMAGE_DIR, OUTPUT_THERMAL_DIR)
    thermal_resize.resize_images_in_folder()
    
    
    
    