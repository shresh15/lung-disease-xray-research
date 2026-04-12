import torch
import torch.optim as optim
import pandas as pd

from config import DATASETS, DEVICE
from datasets.dataset_loader import load_kaggle, load_covid, load_nih
from models.resnet_model import get_model
from training.trainer import train
from training.losses import get_loss
from evaluation.evaluate import evaluate


results = []


def load_train_dataset(name):

    if name == "chest_xray":
        return load_kaggle(DATASETS["kaggle"]["train"])

    if name == "covid":
        return load_covid(DATASETS["covid"]["train"])

    if name == "nih":
        return load_nih(DATASETS["nih"]["train"], DATASETS["nih"]["csv"])


def load_test_dataset(name):

    if name == "chest_xray":
        return load_kaggle(DATASETS["kaggle"]["test"])

    if name == "covid":
        return load_covid(DATASETS["covid"]["test"])

    if name == "nih":
        return load_nih(DATASETS["nih"]["test"], DATASETS["nih"]["csv"])



dataset_names = ["chest_xray", "nih", "covid"]


for train_name in dataset_names:

    for test_name in dataset_names:

        if train_name == test_name:
            continue

        print(f"\n===== {train_name.upper()} → {test_name.upper()} =====")

        train_loader = load_train_dataset(train_name)
        test_loader = load_test_dataset(test_name)

        model = get_model().to(DEVICE)

        criterion = get_loss()

        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        train(model, train_loader, optimizer, criterion)

        metrics = evaluate(model, test_loader)

        results.append({
            "train_dataset": train_name,
            "test_dataset": test_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"]
        })


df = pd.DataFrame(results)

print("\nFinal Results Table")
print(df)

df.to_csv("cross_dataset_results.csv", index=False)