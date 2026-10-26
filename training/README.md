# Training and inference of the segmentation methods

Code used for training the 16 segmentation methods compared in the paper and for generating their predictions. The predictions are probability maps saved as `.npz` files, which are used by [`run_models.py`](../run_models.py) for calculating the metrics.

## Installation

Python 3.13 or later and a CUDA-capable GPU are required. The dependencies are different from those of the metrics code and are listed in [requirements.txt](requirements.txt):

```bash
pip install -r requirements.txt
```

## Data

Each dataset must be split into folders containing the images and the corresponding labels. The paths are set in the configuration files, for instance:

```yaml
train_x_path: '../data/HRF/train/input',
train_y_path: '../data/HRF/train/label',
val_x_path: '../data/HRF/val/input',
val_y_path: '../data/HRF/val/label',
```

## Training

The script `train_all.py` trains all 16 methods on one dataset using the configuration in [configs/train.yml](configs/train.yml). Besides the dataset paths, the following settings must be adapted to each dataset:

```yaml
# Base directory to save all trained models
saved_model_directory: '../data/models/HRF',
# 'RGB' for color images, 'RGBfG' for grayscale datasets (VessMAP, DCA1 and OCTA)
input_space: 'RGB',
# Next multiple of 32 of the image resolution
# DRIVE: 608, STARE: 704, CHASE_DB1: 1024, HRF: 1344, DCA1: 320, OCTA: 416, VessMAP: 256
input_size: [1344, 1344],
```

The remaining hyperparameters (optimizer, learning rate schedule, loss function, and data augmentation) are the ones described in the paper. Run:

```bash
python train_all.py
```

The log of all runs is saved in `train_all.log`.

## Inference

The script `inference_all.py` segments the images in `val_x_path` using the trained models, with the configuration in [configs/inference.yml](configs/inference.yml). The settings `model_path`, `val_x_path`, `val_y_path`, `input_space`, and `input_size` must match the ones used for training. Run:

```bash
python inference_all.py
```

## Acknowledgments

The code is adapted from the [FSG-Net](https://github.com/ZombaSY/FSG-Net-pytorch) repository, with modifications for: 1) automated training and inference of all methods; 2) reduction of the GPU memory usage of some methods to allow training on a single GPU; 3) compilation of the models using `torch.compile`; 4) bug fixes related to image resizing; 5) the inclusion of new methods. 
