import torch
from sklearn.metrics import accuracy_score, classification_report
from config import DEVICE


def evaluate(model, loader):

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    print("Accuracy:", accuracy_score(y_true, y_pred))

    print(classification_report(y_true, y_pred))