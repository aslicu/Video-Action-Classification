# Video-Action-Classification
video action classification using Pytorch R3D-18 model
### Dataset 
You can download the UCF-101 dataset from the official source: [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) Dataset. 
### Directory Structure
workplace/
├── UCF101_Train/
│   └── videos/   # Training videos
├── UCF101_Test/
│   └── videos/   # Testing videos
├── train.txt     # List of video clips for training (with optional semi-supervised labels)
└── test.txt      # List of video clips for testing


### Checkpoints
The trained models, used to produce the numbers in the paper, can be downloaded here.

### Running

```
python main.py --batch_size 32 --clip_len 16 --crop_size 256 --epochs 50 --lr 0.0001

```
###  Acknowledgements
This code is based on [Rethinking Zero-shot Video Classification](https://github.com/bbrattoli/ZeroShotVideoClassification/) repository
