import os
from PIL import Image

def convert_jpg_to_ong(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            jpg_path = os.path.join(folder_path, filename)
            png_path = os.path.join(folder_path, filename[:-4]+'.png')
            
            with Image.open(jpg_path) as img:
                img.save(png_path, 'PNG')
                print(f'Converted: {jpg_path} to {png_path}')
            
            os.remove(jpg_path)
            print(f'Deleted: {jpg_path}')

if __name__ == '__main__':
    folder_path = './datasets/Scene2ver2/256x256_testB/'
    convert_jpg_to_ong(folder_path=folder_path)