import os, numpy as np
import cv2, torch
from torch.utils.data import Dataset
from transforms.transforms import get_transform


def read_split_data(split_path=""):
    videossl_split_doc=open(split_path)
    split_list=videossl_split_doc.readlines()
    i = 0
    for vid in split_list:
        split_list[i] = vid[:-1]
        i = i + 1
    return split_list

def load_split_data(path="",split_path=""):
    folder = path
    fnames, labels = [], []
    video_split = read_split_data(split_path)
    for label in sorted(os.listdir(str(folder))):
        for fname in os.listdir(os.path.join(str(folder), label)):
            if fname  in video_split:
                fnames.append(os.path.join(str(folder), label, fname))
                labels.append(label)
    classes = np.unique(labels)
    return fnames, labels, classes

def load_clips_tsn(fname, clip_len=16, n_clips=1, is_validation=False):
    if not os.path.exists(fname):
        print('Missing: '+fname)
        return []
    # initialize a VideoCapture object to read video data into a numpy array
    capture = cv2.VideoCapture(fname)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_count == 0 or frame_width == 0 or frame_height == 0:
        print('loading error, switching video ...')
        print(fname)
        return []

    total_frames = frame_count #min(frame_count, 300)
    sampling_period = max(total_frames // n_clips, 1)
    n_snipets = min(n_clips, total_frames // sampling_period)
    if not is_validation:
        starts = np.random.randint(0, max(1, sampling_period - clip_len), n_snipets)
    else:
        starts = np.zeros(n_snipets)
    offsets = np.arange(0, total_frames, sampling_period)
    selection = np.concatenate([np.arange(of+s, of+s+clip_len) for of, s in zip(offsets, starts)])

    frames = []
    count = ret_count = 0
    while count < selection[-1]+clip_len:
        retained, frame = capture.read()
        if count not in selection:
            count += 1
            continue
        if not retained:
            if len(frames) > 0:
                frame = np.copy(frames[-1])
            else:
                frame = (255*np.random.rand(frame_height, frame_width, 3)).astype('uint8')
            frames.append(frame)
            ret_count += 1
            count += 1
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        count += 1
    capture.release()
    frames = np.stack(frames)
    total = n_clips * clip_len
    while frames.shape[0] < total:
        frames = np.concatenate([frames, frames[:(total - frames.shape[0])]])
    frames = frames.reshape([n_clips, clip_len, frame_height, frame_width, 3])
    return frames


class VideoDataset(Dataset):
    def __init__(self, fnames, labels, classes, load_clips=load_clips_tsn,
                 clip_len=8, n_clips=1, crop_size=112, is_validation=False):

        self.data = fnames
        self.labels = labels
        self.class_name = classes


        self.clip_len = clip_len
        self.n_clips = n_clips
        self.crop_size = crop_size
        self.is_validation = is_validation

        # prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)
        self.transform = get_transform(self.is_validation, crop_size)
        self.loadvideo = load_clips

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.label_array[idx]
        buffer = self.loadvideo(sample, self.clip_len, self.n_clips, self.is_validation)
        s = buffer.shape
        buffer = buffer.reshape(s[0] * s[1], s[2], s[3], s[4])
        buffer_org = torch.stack([torch.from_numpy(im) for im in buffer], 0)
        buffer = self.transform(buffer_org)
        buffer = buffer.reshape(3, s[0], s[1], self.crop_size, self.crop_size).transpose(0, 1)
        class_name = self.class_name[label]  # Get the class name as text from the label
        return buffer, label, class_name, idx
    def __len__(self):
        return len(self.data)
