"""
人数ごとにiouの平均を求める
"""
import os
import cv2
import re
import numpy as np
from glob import glob


def get_txt_from_subfolder(base_folder, subfolder_name):
    """
    指定されたフォルダ内の連番形式のサブフォルダ
    """
    result = {}
    
    # サブフォルダ名
    subfolder_name = os.path.join(base_folder, subfolder_name)
    
    # サブフォルダを取得
    for folder_name in sorted(os.listdir(subfolder_name)):
        image_folder_path = os.path.join(subfolder_name, folder_name)
        
        # サブフォルダか確認
        if os.path.isdir(image_folder_path):
            # サブフォルダ番号を抽出
            folder_num = folder_name.split("_")[-1]
            
            # テキストファイル名
            txt_file_name = f"iou_{folder_num}.txt"
            txt_file_path = os.path.join(image_folder_path, txt_file_name)
            
            # テキストファイルの存在確認
            if os.path.isfile(txt_file_path):
                result[folder_name] = txt_file_path
    return result

def extract_iou_value(txt_file_path):
    try:
        with open(txt_file_path, "r") as file:
            content = file.read()
            
            match = re.search(r"IoU:\s*([0-9]*\.?[0-9]+)", content)
            if match:
                return float(match.group(1))
    except Exception as e:
        print(f"Eroor reading file {txt_file_path}: {e}")
    return None

def get_mean_IoU_from_subfolder(base_folder, subfolder_name):
    txt_path_list = get_txt_from_subfolder(base_folder, subfolder_name)
    
    txt_files_num = len(txt_path_list)
    iou_sum = 0
    # フォルダごとのiouを取得する
    for folder_name, txt_path in txt_path_list.items():
        iou_value = extract_iou_value(txt_path)
        iou_sum += iou_value
        
    mean_iou = iou_sum / txt_files_num
    print(f"{subfolder_name}:{mean_iou:.6f}")
    return iou_sum, txt_files_num

base_folder = './DCLGAN/instance_seg/rgb2thermal_SimDCL_400/iou_by_people'
subfolder_name_one = 'one_person'
subfolder_name_two = 'two_persons'
subfolder_name_three = 'three_persons'
subfolder_name_four = 'four_persons'

iou_val_one_person, files_num_one_person = get_mean_IoU_from_subfolder(base_folder, subfolder_name_one)
iou_val_two_persons, files_num_two_persons = get_mean_IoU_from_subfolder(base_folder, subfolder_name_two)
iou_val_three_persons, files_num_three_persons = get_mean_IoU_from_subfolder(base_folder, subfolder_name_three)
iou_val_four_persons, files_num_four_persons = get_mean_IoU_from_subfolder(base_folder, subfolder_name_four)

all_iou = iou_val_one_person + iou_val_two_persons + iou_val_three_persons + iou_val_four_persons

all_files_num = files_num_one_person + files_num_two_persons + files_num_three_persons + files_num_four_persons

mean_all_iou = all_iou / all_files_num
print(f"mean IoU:{mean_all_iou:.6f}")