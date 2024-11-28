# Video-Action-Classification
# Dataset 

You can download the UCF-101 dataset from the official source: UCF101 Dataset.

# Directory Structure
workplace/
├── UCF101_Train/
│   └── videos/   # Training videos
├── UCF101_Test/
│   └── videos/   # Testing videos
├── train.txt     # List of video clips for training (with optional semi-supervised labels)
└── test.txt      # List of video clips for testing

train.txt and test.txt should contain the relative paths to the video clips and their corresponding labels. These files can be customized to include labeled or unlabeled video clips for semi-supervised learning.

# Checkpoints
The trained models, used to produce the numbers in the paper, can be downloaded here.

### Running

```
python main.py --batch_size 32 --clip_len 16 --crop_size 256 --epochs 50 --lr 0.0001

```
