import torch
import argparse
from datasets.dataloader import get_dataloader
from models.resnet import get_resnet
from training.train import train_model
from evaluation.evaluate import evaluate

# -------------------------------
# Argument Parser
# -------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--train_dataset", type=str, required=True,
                    help="Dataset to train on (kaggle_pneumonia / nih / covid)")
parser.add_argument("--test_dataset", type=str, required=True,
                    help="Dataset to test on (kaggle_pneumonia / nih / covid)")
parser.add_argument("--batch_size", type=int, default=32)

args = parser.parse_args()

# -------------------------------
# Device
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -------------------------------
# Dataset Paths Mapping
# -------------------------------
DATASET_PATHS = {
    "kaggle_pneumonia": "datasets/kaggle_pneumonia",
    "nih": "datasets/nih",
    "covid": "datasets/covid"
}

train_path = DATASET_PATHS[args.train_dataset]
test_path = DATASET_PATHS[args.test_dataset]

print(f"Training on: {train_path}")
print(f"Testing on: {test_path}")

# -------------------------------
# Load Data
# -------------------------------
train_loader, train_classes = get_dataloader(train_path, args.batch_size)
test_loader, test_classes = get_dataloader(test_path, args.batch_size)

# -------------------------------
# Model
# -------------------------------
model = get_resnet(len(train_classes))

# -------------------------------
# Train
# -------------------------------
model = train_model(model, train_loader, device)

# -------------------------------
# Evaluate
# -------------------------------
evaluate(model, test_loader, device)