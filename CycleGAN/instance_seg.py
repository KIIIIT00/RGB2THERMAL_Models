"""
インスタンスセグメンテーション with yolo11seg
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO

def people_segmentation(model, img_path, color):
    """
    指定した画像に対して人のインスタンスセグメンテーションをする
    """
    results = model(img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if results[0].masks is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        masks = results[0].masks.xy
        for mask, cls in zip(masks, clss):
            if cls == 0:  # "person"クラスの場合
                mask = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))
                
                overlay = np.zeros_like(img, dtype=np.uint8)
                cv2.fillPoly(overlay, [mask], color[:3])  # RGBのみ使用
                img = cv2.addWeighted(img, 0.2, overlay, 1, 0)
        
    return img

def overlay_image(original_image, thermal_image, gen_thermal_image, overlay_num = 3):
    # すべての画像サイズを揃える
    height, width = gen_thermal_image.shape[:2]
    original_image_resized = cv2.resize(original_image, (width, height))
    thermal_image_resized = cv2.resize(thermal_image, (width, height))
    
    # 3つの画像を重ねる
    if overlay_num == 3:
        overlay = cv2.addWeighted(original_image_resized, 0.5, thermal_image_resized, 0.5, 0)
        overlay = cv2.addWeighted(overlay, 0.5, gen_thermal_image, 0.5, 0)
    elif overlay_num == 2:
        overlay = cv2.addWeighted(gen_thermal_image, 0.5, thermal_image_resized, 0.5, 0)
    return overlay

# モデルと画像パスの設定
MODELS_DIR_PATH = '.yolo_models'
MODEL_FILE_NAME = 'yolo11x-seg.pt'
MODEL_FILE_PATH = os.path.join(MODELS_DIR_PATH, MODEL_FILE_NAME)

IMG_NUM = 87
ORIGINAL_IMG_PATH = f'./CSTGAN/datasets/Scene2ver2/testA/rgb_{IMG_NUM}.jpg'
THERMAL_IMG_PATH = f'./CSTGAN/datasets/Scene2ver2/testB/thermal_{IMG_NUM}.jpg'
GEN_THERMAL_IMG_PATH = f'./CycleGAN/results/Scene2ver2_500_lambda10.5/Scene2ver2_500_lambda10.5/test_latest/images/rgb_{IMG_NUM}_fake.jpg'

# 色設定
mask_color_dict = {
    "original": (0, 255, 0),  # 緑
    "thermal": (0, 0, 255),  # 青
    "gen_thermal": (255, 0, 0)  # 赤
}

# モデルの読み込み
model = YOLO(MODEL_FILE_PATH)

# セグメンテーションと画像のオーバーレイ
original_img = people_segmentation(model, ORIGINAL_IMG_PATH, mask_color_dict["original"])
thermal_img = people_segmentation(model, THERMAL_IMG_PATH, mask_color_dict["thermal"])
gen_thermal_img =  people_segmentation(model, GEN_THERMAL_IMG_PATH, mask_color_dict["gen_thermal"])

# 画像の重ね合わせ
img = overlay_image(original_img, thermal_img, gen_thermal_img, 2)

# 結果を表示
cv2.imshow("Segmentation", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()