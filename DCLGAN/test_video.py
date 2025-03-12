"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import cv2
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
from tqdm import tqdm
import glob

def save_frames_and_get_fps(video_path, output_folder):
    # video file name
    video_file_name = os.path.splitext(os.path.basename(video_path))[0]
    # output folder path
    output_folder_path = os.path.join(output_folder, video_file_name)
    
    # load video file
    video_cap = cv2.VideoCapture(video_path)
    print(f"VideoPath:{video_path}")
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS:{fps}")
    
    # check if video file is opened correctly
    if not video_cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return fps, None
    
    # create output folder
    os.makedirs(output_folder_path, exist_ok=True)
    
    os.makedirs(os.path.join(output_folder_path, 'testA'), exist_ok=True)
    os.makedirs(os.path.join(output_folder_path, 'testB'), exist_ok=True)
    os.makedirs(os.path.join(output_folder_path, 'trainA'), exist_ok=True)
    os.makedirs(os.path.join(output_folder_path, 'trainB'), exist_ok=True)
    
    # save frames to output folder
    frame_count = 0
    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            break
        frame_name = f"{video_file_name}_{frame_count}.jpg"
        cv2.imwrite(os.path.join(os.path.join(output_folder_path, 'testA'), frame_name), frame)
        cv2.imwrite(os.path.join(os.path.join(output_folder_path, 'testB'), frame_name), frame)
        cv2.imwrite(os.path.join(os.path.join(output_folder_path, 'trainA'), frame_name), frame)
        cv2.imwrite(os.path.join(os.path.join(output_folder_path, 'trainB'), frame_name), frame)
        
        frame_count += 1
    
    video_cap.release()
    return output_folder_path, fps

def create_video_from_frames(input_folder, output_video_path, fps):
    # input folder path
    input_folder_path = os.path.join(input_folder, 'fake_B')
    print(os.path.isdir(input_folder_path))
    print(input_folder_path)
    input_folder_imgs = glob.glob(os.path.join(input_folder_path, "*.png"))
    
    print("Checking path:", output_images_model_path)
    print("Path exists?", os.path.exists(output_images_model_path))
    print("Files in directory:", os.listdir(output_images_model_path) if os.path.exists(output_images_model_path) else "Directory not found")
    
    # フレームのリストを取得
    frame_files = sorted(input_folder_imgs, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    print("frames_name", frame_files[0])
    if len(frame_files) == 0:
        print("フレーム画像が見つかりませんでした。")
        return
    print("frame len:", len(frame_files))
    # 最初のフレームから動画のサイズを取得
    first_frame = cv2.imread(frame_files[0])
    height, width, layers = first_frame.shape
    print("height", height)
    print("width", width)
    if first_frame is None:
        print(f"Error: Could not read the first frame {frame_files[0]}")
        return
    # 動画ライターを設定
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 出力形式はmp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    print("fps:",fps)
    # フレームを1つずつ動画に追加
    for frame_path in tqdm(frame_files):
        #frame_path = os.path.join(input_folder, frame_file)
        frame = cv2.imread(frame_path)
        # フレームの読み込みが失敗した場合をチェック
        if frame is None:
            print(f"Error: Could not read frame {frame_path}")
            continue
        # フレームサイズが一致しない場合はリサイズ
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        out.write(frame)
    out.release()
    print(f"動画の作成が完了しました: {output_video_path}")

if __name__ == '__main__':
    
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.num_test = 6000 # test count
    # video settings
    input_video_path = opt.dataroot
    # video split frames
    output_frames_path = './DataSet/'
    video_file_name = os.path.splitext(os.path.basename(input_video_path))[0]
    # get fps and dataroot path
    dataroot_path, fps = save_frames_and_get_fps(input_video_path, output_frames_path)
    
    # update opt
    setattr(opt, 'dataroot', dataroot_path)
    print("After opt:",opt)
    
    results_dir = opt.results_dir
    model_name = opt.name
    
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)      # create a model given opt.model and other options
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    for i, data in enumerate(dataset):
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, width=opt.display_winsize)
    
    # video_file_name = 'AVG-TownCentre-raw'
    # results_dir = f'./DCLGAN/results/{video_file_name}'
    # model_name = 'rgb2thermal_SimDCL_400'
    # fps = 2.5
    # output images from model to video
    output_images_model_path = os.path.join(results_dir, model_name+'/test_latest/images')
    
    output_video_path = os.path.join(results_dir, model_name+f'/test_latest/thermal_{video_file_name}.mp4')
    print("output_video path:", output_video_path)
    create_video_from_frames(output_images_model_path, output_video_path, fps)
    # webpage.save()  # save the HTML
