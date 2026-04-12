import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from datasets.preprocessing import get_transforms
from config import BATCH_SIZE
transform = get_transforms()

# 1️⃣ Kaggle Chest X-ray Loader
# --------------------------------

def load_chest_xray(path):

    dataset = ImageFolder(
        root=path,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    return loader


# --------------------------------
# 2️⃣ COVID Dataset Loader
# --------------------------------

class CovidDataset(Dataset):

    def __init__(self, root_dir):

        self.samples = []

        for label in os.listdir(root_dir):

            label_path = os.path.join(root_dir, label)

            if not os.path.isdir(label_path):
                continue

            for img in os.listdir(label_path):

                img_path = os.path.join(label_path, img)

                label_lower = label.lower()

                if label_lower == "normal":
                    mapped_label = 0

                elif label_lower == "pneumonia":
                    mapped_label = 1

                elif label_lower == "covid_19":
                    mapped_label = 1   # covid pneumonia

                else:
                    continue

                self.samples.append((img_path, mapped_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        image = transform(image)

        return image, label


def load_covid(path):

    dataset = CovidDataset(path)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    return loader


# --------------------------------
# 3️⃣ NIH Dataset Loader
# --------------------------------

class NIHDataset(Dataset):

    def __init__(self, img_dir, csv_file):

        self.img_dir = img_dir
        df = pd.read_csv(csv_file)

        self.samples = []

        for _, row in df.iterrows():

            img_name = row["Image Index"]
            labels = row["Finding Labels"]

            img_path = os.path.join(img_dir, img_name)

            if not os.path.exists(img_path):
                continue

            if "Pneumonia" in labels:
                mapped_label = 1
            else:
                mapped_label = 0

            self.samples.append((img_path, mapped_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        image = transform(image)

        return image, label


def load_nih(img_dir, csv_file):

    dataset = NIHDataset(img_dir, csv_file)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    return loader