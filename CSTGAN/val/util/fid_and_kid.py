import numpy as np
import torch
import os
from PIL import Image
from torchvision import models, transforms
from scipy.stats import gaussian_kde

class ImageDistatnceCalculator:
    def __init__(self, image_size=(256, 256), num_samples=1000, device='cuda'):
        self.image_size = image_size
        self.num_samples = num_samples
        self.device = device
        self.vgg_model = models.vgg16(pretrained=True).to(device).features.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize(self.image_size, transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _get_activation(self, images):
        """ Get the activations from the model """
        images = images.to(next(self.vgg_model.parameters()).device)
        with torch.no_grad():
            return self.vgg_model(images).view(images.size(0), -1)
    
    def calculate_fid(self, real_image, fake_image):
        """ Calculate FID between real and fake image"""
        real_image = torch.stack([self.preprocess(real_image)]).to('cuda')
        fake_image = torch.stack([self.preprocess(fake_image)]).to('cuda')
        
        real_activations = self._get_activation(real_image)
        fake_activations = self._get_activation(fake_image)
        
        mu1, sigma1 = real_activations.mean(dim=0), torch.cov(real_activations.T)
        mu2, sigma2 = fake_activations.mean(dim=0), torch.cov(fake_activations.T)
        
        fid_value = (mu1 - mu2).dot(mu1 - mu2) + torch.trace(sigma1 + sigma2 - 2 * torch.sqrt(sigma1 @ sigma2))
        return fid_value.item()
    
    def calculate_kid(self, real_image, fake_image):
        """Calculate KID between real and fake image."""
        real_image = torch.stack([self.preprocess(real_image)]).to('cuda')
        fake_image = torch.stack([self.preprocess(fake_image)]).to('cuda')
        
        real_activations = self._get_activation(real_image)
        fake_activations = self._get_activation(fake_image)
        
        if len(real_activations) > self.num_samples:
            real_activations = real_activations[torch.randperm(len(real_activations))[:self.num_samples]]
        if len(fake_activations) > self.num_samples:
            fake_activations = fake_activations[torch.randperm(len(fake_activations))[:self.num_samples]]
        
        real_kde = gaussian_kde(real_activations.cpu().numpy().T)
        fake_kde = gaussian_kde(fake_activations.cpu().numpy().T)
        
        kid_value = np.mean(real_kde.evaluate(fake_activations.cpu().numpy().T)) - np.mean(fake_kde.evaluate(real_activations.cpu().numpy().T))
        return kid_value
    
    def calculate_distances(self, real_directory, fake_directory):
        """Load images from directories and calculate both FID and KID."""
        paired_images = self.load_images_from_directory(real_directory, fake_directory)
        data_size = len(paired_images)
        
        fid_scores = 0
        kid_scores = 0
        
        for real_image, fake_image in paired_images:
            fid_score = self.calculate_fid(real_image, fake_image)
            fid_scores += fid_score
            
            kid_score = self.calculate_kid(real_image, fake_image)
            kid_scores += kid_score
        
        fid_means = fid_scores / data_size
        kid_means = kid_scores / data_size
        return fid_means, kid_means
    
    def load_images_from_directory(self, real_directory, fake_directory):
        real_images = self.load_images_from_directory_by_type(real_directory, 'real')
        fake_images = self.load_images_from_directory_by_type(fake_directory, 'fake')
        
        paired_images = list(zip(real_images, fake_images))
        
        if not paired_images:
            print("No paired images found.")
        
        return paired_images
    
    def load_images_from_directory_by_type(self, directory, real_or_fake):
        images = []
        
        if real_or_fake not in ['real', 'fake']:
            raise ValueError("real_or_fake argument must be 'real' or 'fake'")
        
        # 指定された条件に基づき、対応するファイルのみを取得し、数値部分でソート
        if real_or_fake == 'real':
            file_list = sorted(
                [f for f in os.listdir(directory) if f.startswith('thermal_') and f.endswith('.png')],
                key=lambda x: int(x.split('_')[1].split('.')[0])
            )
        elif real_or_fake == 'fake':
            file_list = sorted(
                [f for f in os.listdir(directory) if f.endswith('_fake.png')],
                key=lambda x: int(x.split('_')[1])
            )
        
        for filename in file_list:
            image_path = os.path.join(directory, filename)
            images.append(Image.open(image_path).convert('RGB'))
            print(f"{real_or_fake.capitalize()}: {image_path}")
        
        return images
    
if __name__ == '__main__':
    MODEL_NAME = 'Scene2ver2_100_lambda10'
    INPUT_FAKE_FOLDER = f'./results/{MODEL_NAME}/{MODEL_NAME}/test_latest/images/'
    INPUT_REAL_FOLDER = './datasets/Scene2ver2/256x256_testB/'
    
    distance_calculator = ImageDistatnceCalculator()
    fid, kid = distance_calculator.calculate_distances(INPUT_REAL_FOLDER, INPUT_FAKE_FOLDER)
    print(f"FID:{fid}, KID:{kid}")