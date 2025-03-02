import os
import torch
import lpips
from PIL import Image
import torchvision.transforms as transforms

class LPIPSEvaluator:
    def __init__(self, real_directory, fake_directory, model='alex'):
        self.real_directory = real_directory
        self.fake_directory = fake_directory
        self.lpips_model = lpips.LPIPS(net=model)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 必要に応じてリサイズ
            transforms.ToTensor()
        ])
    
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
    
    def load_image(self, image):
        return self.transform(image).unsqueeze(0)
    
    def calculate_lpips(self):
        paired_images = self.load_images_from_directory(self.real_directory, self.fake_directory)
        data_size = len(paired_images)
        lpips_distances = 0
        
        for real_image, fake_image in paired_images:
            real_image = self.load_image(real_image)
            fake_image = self.load_image(fake_image)
            
            with torch.no_grad():
                lpips_distance = self.lpips_model(real_image, fake_image).item()
                
            lpips_distances += lpips_distance
            
        return lpips_distances / data_size
    
if __name__ == '__main__':
    print(dir(lpips))
    MODEL_NAME = 'Scene2ver2_100_lambda10'
    INPUT_FAKE_FOLDER = f'./results/{MODEL_NAME}/{MODEL_NAME}/test_latest/images/'
    INPUT_REAL_FOLDER = './datasets/Scene2ver2/256x256_testB/'
    
    evaluator = LPIPSEvaluator(INPUT_REAL_FOLDER, INPUT_FAKE_FOLDER, 'alex')
    mean_distances = evaluator.calculate_lpips()
    print(f"LPIPS Distance: {mean_distances}")

