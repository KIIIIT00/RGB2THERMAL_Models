# RGB2THERMAL_Models

## Description   
[RGB2THERMAL](https://github.com/KIIIIT00/RGBtoTHERMAL) のプロジェクトを用いて採集したデータセットを使用し，様々な機械学習モデルを学習させる．

## ディレクトリ構造
```
.
├── CSTGAN 
├── CycleGAN 
├── DCLGAN //CycleGANとpix2pixの基盤モデル
├── MUNIT
├── yolo_models //YOLOv11のセグメンテーションファイルを置くフォルダ
├── utils // 使いまわすクラス
├── .gitignore
├── Dockerfile
├── environment.yml
├── LICENSE
└── README.md
```
## How to run   
1. **CycleGAN**
### train
```
$ python train.py --dataroot ./datasets/inputs --n_epochs 100 --name model_name --display_id 0 --gpu_ids 0
```

### test
```
$ python test.py --dataroot ./datasets/inputs --name model_name --results_dir ./results/outputs --model test --no_dropout
```

2. **CSTGAN**
### train
```
$ python train_tensorboard.py --dataroot ./datasets/inputs --n_epochs 100 --name model_name --display_id 0 --gpu_ids 0
```

### test
```
$ python test.py --dataroot ./datasets/inputs --name model_name --result_dir ./results/outputs --moddel test --no_dropout
```

3. **DCLGAN**
### train
```
$ python train.py --dataroot ./datasets/inputs --n_epochs 100 --name model_name --model simdcl
```

### test
```
$ python test.py --dataroot ./datasets/inputs --name model_name
```

4. **MUNIT**
### train
```
$ python train.py --config ./config/model_settings.yaml
```

### test
```
$ python test_multi.py --config ./config/model_settings.yaml --input ./datasets/inputs --output_folder ./results/outputs --checkpoint ./outputs/model_name/checkpoints/gen_000000.pt --a2b 1
```
