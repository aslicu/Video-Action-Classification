# Video-Action-Classification
## Dataset Preparation

You can download the UCF-101 dataset from the official source: UCF101 Dataset.

## Directory Structure
train.txt and test.txt should contain the relative paths to the video clips and their corresponding labels. These files can be customized to include labeled or unlabeled video clips for semi-supervised learning.

## Checkpoints
The trained models, used to produce the numbers in the paper, can be downloaded here.

## Running
'''
python main.py --batch_size 32 --clip_len 16 --crop_size 256 --epochs 50 --lr 0.0001
'''


