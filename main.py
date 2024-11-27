import argparse
import os
from transforms.transforms import batch2gif
from video_data.video_dataset import *
from models.model import *
import torch
from torch.utils.data import Dataset, DataLoader

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train a 3D video classification model.")
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and testing.')
    parser.add_argument('--clip_len', type=int, default=8, help='Number of frames in a video clip.')
    parser.add_argument('--crop_size', type=int, default=224, help='Crop size for video frames.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer.')
    return parser.parse_args()

# Parse user arguments
args = parse_args()

# Utility function for creating directories
def create_dir(base_dir, sub_dir):
    """Creates and returns the absolute path for a subdirectory under base_dir."""
    path = os.path.join(base_dir, sub_dir)
    os.makedirs(path, exist_ok=True)
    return path

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get the base directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define base directories
workplace_dir = create_dir(script_dir, 'workplace')
gif_save_base_dir = create_dir(script_dir, 'mygifs')
save_dir = create_dir(script_dir, 'model_checkpoints')

# Define paths relative to the workplace directory
train_video_path = os.path.join(workplace_dir, 'UCF101_Train/videos/')
test_video_path = os.path.join(workplace_dir, 'UCF101_Test/videos/')
train_split_path = os.path.join(workplace_dir, 'train.txt')
test_split_path = os.path.join(workplace_dir, 'test.txt')

def get_video_datasets(clip_len=args.clip_len, n_clips=1, seed=42):
    np.random.seed(seed)  # Set the random seed for reproducibility
    fnames_test, labels_test, classes_test = load_split_data(path=test_video_path, split_path=test_split_path)
    fnames_train, labels_train_labeled, classes_train_labeled = load_split_data(train_video_path, train_split_path)

    train_dataset = VideoDataset(
        fnames_train, labels_train_labeled, classes_train_labeled,
        clip_len=clip_len, n_clips=n_clips, crop_size=args.crop_size, is_validation=False
    )

    test_dataset = VideoDataset(
        fnames_test, labels_test, classes_test,
        clip_len=clip_len, n_clips=1, crop_size=args.crop_size, is_validation=True
    )
    return {'training': train_dataset, 'testing': test_dataset}

datasets = get_video_datasets()

# Instantiate your custom model
num_classes = 101  # Adjust this based on your dataset
model = VideoR3D18(num_classes)
model.to(device)

model_checkpoint = os.path.join(save_dir, 'model_best.pth')

if os.path.exists(model_checkpoint):
    checkpoint = torch.load(model_checkpoint)
    model.load_state_dict(checkpoint, strict=False)
    print(f"Loaded model from {model_checkpoint}")
else:
    print(f"No checkpoint found at {model_checkpoint}, training from scratch.")

# Define the optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

# Define the data iterator and number of epochs
train_dataset = datasets['training']
test_dataset = datasets['testing']

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

for epoch in range(args.epochs):
    print(f"Epoch {epoch + 1}/{args.epochs}")
    total_loss = 0.0
    total_batches = 0
    correct_predictions = 0
    total_samples = 0
    model.train()
    for i, (X, l, _, j) in enumerate(train_loader):  # Iterate over batches from DataLoader
        X, l = X.squeeze(1).to(device), l.to(device)
        if (i % 10) == 0:
            savepath = os.path.join(gif_save_base_dir, f"{epoch + 1}_{i}")
            batch2gif(X[0], l[0], savepath)
        outputs, _ = model(X)
        l_tensor = l.long()
        loss = criterion(outputs, l_tensor)
        total_loss += loss.item()
        total_batches += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += l_tensor.size(0)
        correct_predictions += (predicted == l_tensor).sum().item()
    avg_loss = total_loss / total_batches
    accuracy = (correct_predictions / total_samples) * 100
    print(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Testing
    model.eval()
    total_loss = 0.0
    total_batches = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for i, (X, l, _, j) in enumerate(test_loader):
            X = X.squeeze(1)
            X, l = X.to(device), l.to(device)

            outputs, _ = model(X)
            l_tensor = l.long()
            loss = criterion(outputs, l_tensor)

            total_loss += loss.item()
            total_batches += 1

            _, predicted = torch.max(outputs.data, 1)
            total_samples += l_tensor.size(0)
            correct_predictions += (predicted == l_tensor).sum().item()

    avg_loss = total_loss / total_batches
    accuracy = (correct_predictions / total_samples) * 100
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    model_save_path = os.path.join(save_dir, f"baseliner3d_{epoch + 1}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")
