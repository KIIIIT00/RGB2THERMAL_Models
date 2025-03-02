"""
Yolov11のセグメンテーションを用いた
セマンティックセグメンテーション
凡例を付ける
IoUを計算する
"""
import os
import cv2
import re
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from glob import glob

def people_segmentation(model, img_path, color):
    """
    指定した画像に対して人のインスタンスセグメンテーションをする
    """
    results = model(img_path)
    # 画像読み込み
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    
    # インスタンスセグメンテーション結果を可視化
    if results[0].masks is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        masks = results[0].masks.xy
        for mask, cls in zip(masks, clss):
            if cls == 0:
                mask = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))
                bgr_color = (color[2], color[1], color[0])
                # overlay = np.zeros_like(img, dtype=np.uint8)
                overlay = img.copy()
                cv2.fillPoly(overlay, [mask], bgr_color)  # RGBのみ使用
                cv2.fillPoly(mask_img, [mask], 1)
                img = cv2.addWeighted(img, 0.8, overlay, 0.2, 0)
        
    return img, mask_img

def overlay_image(original_image, thermal_image, gen_thermal_image, overlay_num = 2, image_size=(256,256), overlay_opt='RandT'):
    """
    画像の重ね合わせを行う
    """
    height, width = image_size
    original_image_resized = cv2.resize(original_image, (width, height))
    thermal_image_resized = cv2.resize(thermal_image, (width, height))
    gen_thermal_image_resized = cv2.resize(gen_thermal_image, (width, height))
    
    # 3つの画像を重ねる
    if overlay_num == 3:
        overlay_image = cv2.addWeighted(original_image_resized, 0.5, thermal_image_resized, 0.5, 0)
        overlay_image = cv2.addWeighted(overlay_image, 0.5, gen_thermal_image_resized, 0.5, 0)
    elif overlay_num == 2:
        if overlay_opt == 'RandT':
            overlay_image = cv2.addWeighted(original_image_resized, 0.5, thermal_image_resized, 0.5, 0)
        elif overlay_opt == 'TandG':
            overlay_image = cv2.addWeighted(thermal_image_resized, 0.5, gen_thermal_image_resized, 0.5, 0)
    return overlay_image

def save_overlay_with_legend(img, colors, output_file_path, overlay_opt):
    """
    Overlay画像を凡例付きで保存
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(img)
    
    colors_copy = colors.copy()
    if overlay_opt == 'RandT':
        colors_copy.pop("Generated Thermal")
    elif overlay_opt == 'TandG':
        colors_copy.pop("Original")
    # 凡例の作成
    legend_patches = [
        plt.Line2D([0], [0], color=np.array(color[:3]) / 255, lw=4, label=label)
        for label, color in colors_copy.items()
    ]
    ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.2, 1))
    ax.axis('off')  # 軸を非表示
    plt.tight_layout()
    plt.savefig(output_file_path+'.png', bbox_inches='tight', dpi=500) # ラスタ形式
    plt.savefig(output_file_path+'.svg', bbox_inches='tight') #ベクタ形式
    plt.close()

def calculate_iou(thermal_mask, gen_thermal_mask, image_size = (256, 256)):
    """
    IoUの計算をする
    
    Parameters:
        thermal_mask:赤外線画像に対して人のセマンティックセグメンテーションを行った時のマスク画像
        gen_thermal_mask:生成した赤外線画像に対して人のセマンティックセグメンテーションを行った時のマスク画像
        
    Return:
        iou
    """
    thermal_mask = cv2.resize(thermal_mask, image_size)
    gen_thermal_mask = cv2.resize(gen_thermal_mask, image_size)
    # intersection
    intersection = np.logical_and(thermal_mask, gen_thermal_mask)
    
    # Union
    union = np.logical_or(thermal_mask, gen_thermal_mask)
    
    # unionのピクセル数が0の場合
    if np.sum(union) == 0:
        return 0.0
    
    # IoU計算
    iou = np.sum(intersection) / np.sum(union)
    return iou

def save_iou_text(save_text_path, iou):
    """
    iouをテキストファイルに保存する
    """
    with open(save_text_path, "w") as file:
        file.write(f"IoU: {iou:.6f}")
    print(f"Saved iou {save_text_path}")
    
def process_images_in_folder(folder_paths, model, mask_color_dict, output_dir):
    """
    フォルダ内の全ての画像に対して処理を行う
    """
    # フォルダパスを展開
    original_folder, thermal_folder, gen_thermal_folder = folder_paths
    print(original_folder)
    print(os.path.isdir(original_folder))
    # 出力フォルダの作成
    os.makedirs(output_dir, exist_ok=True)
    
    #ファイル名の正規表現
    file_pattern = re.compile(r"rgb_(\d+)\.png")
    
    original_images = sorted(glob(os.path.join(original_folder, "rgb_*.png")))
    for original_path in original_images:
        match = file_pattern.search(os.path.basename(original_path))
        if not match:
            print(f"{original_path}は，rgb_().pngとマッチしませんでした")
            continue
        idx = match.group(1)
        
        # 写真の番号ごとにフォルダ作成
        os.makedirs(os.path.join(output_dir, f"image_{idx}"), exist_ok=True)
        
        # 同じインデックスの対応する画像を取得
        thermal_path = os.path.join(thermal_folder, f"rgb_{idx}.png")
        gen_thermal_path = os.path.join(gen_thermal_folder, f"rgb_{idx}.png")
        print("---------------------")
        print("original path:", original_path)
        print("thermal path:", thermal_path)
        print("gen_thermal path:", gen_thermal_path)
        print("---------------------")
        
        print("Gen Thermal color:", mask_color_dict["gen_thermal"])
        
        # 画像の読み込みとセグメンテーション
        original_img, original_mask = people_segmentation(model, original_path, mask_color_dict["original"])
        thermal_img, thermal_mask = people_segmentation(model, thermal_path, mask_color_dict["thermal"])
        gen_thermal_img, gen_thermal_mask = people_segmentation(model, gen_thermal_path, mask_color_dict["gen_thermal"])
        
        # overlay画像生成
        overlay_opt_RT = 'RandT'
        overlay_img_RT = overlay_image(original_img, thermal_img, gen_thermal_img, overlay_num=2, overlay_opt=overlay_opt_RT)
        overlay_opt_TG = 'TandG'
        overlay_img_TG = overlay_image(original_img, thermal_img, gen_thermal_img, overlay_num=2, overlay_opt=overlay_opt_TG)
        
        # 凡例付きオーバーレイ画像の保存
        legend_dict = {
            "Original" : mask_color_dict["original"],
            "Thermal" : mask_color_dict["thermal"],
            "Generated Thermal" : mask_color_dict["gen_thermal"]
        }
        
        # iou計算
        iou_val = calculate_iou(thermal_mask, gen_thermal_mask)
        
        # 保存ファイル名
        output_subfolder = os.path.join(output_dir, f"image_{idx}")
        
        seg_original_path = os.path.join(output_subfolder, f"original_seg_{idx}.jpg")
        seg_thermal_path = os.path.join(output_subfolder, f"thermal_seg_{idx}.jpg")
        seg_ge_thermal_path = os.path.join(output_subfolder, f"gen_thermal_seg_{idx}.jpg")
        
        overlay_image_RT_path = os.path.join(output_subfolder, f"overlay_{idx}_RT.jpg")
        overlay_image_TG_path = os.path.join(output_subfolder, f"overlay_{idx}_TG.jpg")
        
        overlay_legend_RT_path = os.path.join(output_subfolder, f"overlay_with_legend_{idx}_RT")
        overlay_legend_TG_path = os.path.join(output_subfolder, f"overlay_with_legend_{idx}_TG")
        
        thermal_mask_path = os.path.join(output_subfolder, f"mask_thermal_{idx}.jpg")
        gen_thermal_mask_path = os.path.join(output_subfolder, f"mask_gen_thermal_{idx}.jpg")
        
        # iou保存ファイル
        iou_text_path = os.path.join(output_subfolder, f"iou_{idx}.txt")
        # iou保存
        save_iou_text(iou_text_path, iou_val)
        
        # 結果画像を保存
        cv2.imwrite(seg_original_path,original_img)
        cv2.imwrite(seg_thermal_path, thermal_img)
        cv2.imwrite(seg_ge_thermal_path, gen_thermal_img)
        cv2.imwrite(overlay_image_RT_path, overlay_img_RT)
        cv2.imwrite(overlay_image_TG_path, overlay_img_TG)
        # マスク画像保存
        cv2.imwrite(thermal_mask_path, thermal_mask*255)
        cv2.imwrite(gen_thermal_mask_path, gen_thermal_mask*255)
        
        save_overlay_with_legend(overlay_img_RT, legend_dict, overlay_legend_RT_path, overlay_opt_RT)
        save_overlay_with_legend(overlay_img_TG, legend_dict, overlay_legend_TG_path, overlay_opt_TG)
        
        base_name = os.path.basename(original_path)
        base_name_no_ext = os.path.splitext(base_name)[0]
        print(f"Saved: {base_name_no_ext}_results")
    
    print("Finished processing images")
# モデルと画像パスの設定
MODELS_DIR_PATH = '.yolo_models'
MODEL_FILE_NAME = 'yolo11x-seg.pt'
MODEL_FILE_PATH = os.path.join(MODELS_DIR_PATH, MODEL_FILE_NAME)

pretrained_model_name = 'rgb2thermal_SimDCL_400'
FOLDER_PATHS = (
    f"./DCLGAN/results/{pretrained_model_name}/test_latest/images/real_A",
    f"./DCLGAN/results/{pretrained_model_name}/test_latest/images/real_B",
    f"./DCLGAN/results/{pretrained_model_name}/test_latest/images/fake_B"
)

OUTPUT_DIR = f'./DCLGAN/instance_seg/{pretrained_model_name}/'

# 色設定
mask_color_dict = {
    "original": (0, 255, 0),  # green
    "thermal": (0, 0, 255),  # blue
    "gen_thermal": (255, 0, 0)  # red
}

# モデルの読み込み
model = YOLO(MODEL_FILE_PATH).to('cpu')

# フォルダの全ての画像に対して処理を実行
process_images_in_folder(FOLDER_PATHS, model, mask_color_dict, OUTPUT_DIR)
