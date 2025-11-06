import torch
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import logging

from lab2.mnist import MNISTDataset, collate_fn
from lab2.lenet import LeNet

logging.basicConfig(
    level=logging.INFO,                                   
    format="%(asctime)s - %(levelname)s - %(message)s",    
    datefmt="%H:%M:%S",                                   
    handlers=[
        logging.StreamHandler(),                          
        logging.FileHandler("training.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using {device} device")

EPOCHS = 10

batch_size = 32
learning_rate = 0.01


def evaluate(dataloader, model):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        with tqdm(desc=f"Evaluating", unit="it", total=len(dataloader)) as pbar:
            for item in dataloader:
                image, label = item["image"].to(device), item["label"].to(device)
                logits = model(image)
                predicted = logits.argmax(dim=-1).long()

                preds.extend(predicted.cpu().tolist())
                labels.extend(label.cpu().tolist())

                pbar.update()

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

def train(dataloader, model, loss_fn, optimizer, epochs) -> None:
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for it, item in enumerate(dataloader):
                images, labels = item["image"].to(device), item["label"].to(device)

                # forward
                pred = model(images)
                loss = loss_fn(pred, labels)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                avg_loss = running_loss / (it + 1)

                pbar.set_postfix(loss=f"{avg_loss:.4f}")
                pbar.update()

        logger.info(f"Epoch {epoch+1}/{epochs} finished | Average loss: {avg_loss:.4f}")
        print(f"Epoch {epoch+1}/{epochs} finished | Average loss: {avg_loss:.4f}")

def main(task):
    loss_fn = torch.nn.CrossEntropyLoss()

    if task not in [1,2,3,4]:
        print(f'Invalid task id: {task}')
        return

    if task == 1:
        mnist_train_image_path = "/kaggle/input/mnist-data/train-images.idx3-ubyte"
        mnist_train_label_path = "/kaggle/input/mnist-data/train-labels.idx1-ubyte"
        mnist_test_image_path = "/kaggle/input/mnist-data/t10k-images.idx3-ubyte"
        mnist_test_label_path = "/kaggle/input/mnist-data/t10k-labels.idx1-ubyte"
        
        mnist_train_dataset = MNISTDataset(mnist_train_image_path, mnist_train_label_path)
        mnist_test_dataset = MNISTDataset(mnist_test_image_path, mnist_test_label_path)
        mnist_train_dataloader = DataLoader(mnist_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        mnist_test_dataloader = DataLoader(mnist_test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        
        model_1 = LeNet().to(device)
        optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=learning_rate)

        logger.info("Training LeNet model")
        train(mnist_train_dataloader, model_1, loss_fn, optimizer_1, EPOCHS)
        metrics_1 = evaluate(mnist_test_dataloader, model_1)
        logger.info(f"Metrics for LeNet model: {metrics_1}")
        print(f"Metrics for LeNet model: {metrics_1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True, help="Select model: 1=LeNet, 2=GoogLeNet, 3=ResNet18, 4=PretrainedResNet")
    args = parser.parse_args()

    main(args.task)