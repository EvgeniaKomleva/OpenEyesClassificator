import os
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet34_Weights,Wide_ResNet50_2_Weights, ResNet50_Weights
import wandb


# ________DATA_______________________________________________________________________________
weights = ResNet34_Weights.IMAGENET1K_V1
train_transform=weights.transforms()
print(train_transform)


BATCH_SIZE = 128
DATA_DIR = './'

dataset = torchvision.datasets.ImageFolder(root=os.path.join(DATA_DIR, 'eyes'), transform=train_transform)
train_set,val_set = torch.utils.data.random_split(dataset, [0.85, 0.15])

train_loader = DataLoader(train_set, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True,  
                          num_workers=0)

val_loader = DataLoader(val_set, 
                        batch_size=BATCH_SIZE, 
                        shuffle=True, 
                        num_workers=0) 




# ________TRAIN_FUNCTION_______________________________________________________________________________

                    
def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_index = np.argmin(np.abs(fnr - fpr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    thresh = thresholds[eer_index]
    return eer, thresh
    

def evaluate(model, loader, criterion, device, ckpt_path=None):
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path))

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    val_labels = []
    val_probs = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            val_labels += labels.cpu().numpy().tolist()
            val_probs += probs[:, 1].cpu().numpy().tolist()

    eer, _ = compute_eer(val_labels, val_probs)
    accuracy = correct / total

    return val_loss, accuracy, eer


def train(model, epoch_num, optimizer, criterion, train_loader, val_loader, ckpt_save_path, lr_scheduler=None):
    
    wandb.watch(model, criterion, log="all", log_freq=1)
    min_val_eer = np.inf

    for epoch in range(epoch_num):

        train_loss = 0.0
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        val_loss, val_accuracy, val_eer = evaluate(model, val_loader, criterion, device)

        wandb.log({"epoch": epoch, "loss": val_loss, 'val_accuracy': val_accuracy, 'val_eer': val_eer})

        print(f"Epoch {epoch}: train loss = {train_loss:.4f}")
        print(f"val loss = {val_loss:.4f}, val accuracy = {val_accuracy:.4f}, val eer = {val_eer:.4f}")

        if lr_scheduler is not None:
            lr_scheduler.step()

        if val_eer < min_val_eer or val_eer < 0.02:
            min_val_eer = val_eer
            torch.save(model.state_dict(), ckpt_save_path)
            print(f"Saving weights with val accuracy = {val_accuracy:.4f}, val eer = {val_eer:.4f}")

    return min_val_eer


# ________TRAIN_MODEL_______________________________________________________________________________

wandb.init(project="open_eyes")
resnet = models.resnet34(pretrained=True)

# Define new Linears layers
class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.fc1 = torch.nn.Linear(resnet.fc.in_features, 256)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(256, 64)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MyNet()
print(model)

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),  lr=0.0001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)

train(model, 100, optimizer,criterion, train_loader, val_loader, './resnet34.pth', lr_scheduler=scheduler)
