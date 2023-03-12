import os
import shutil
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torchvision.models as models
import wandb

# ________DATA_______________________________________________________________________________
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomRotation(degrees=(-45, 45)),
    transforms.RandomPerspective(distortion_scale=0.7, p=1, interpolation=2, fill=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4976, 0.4976, 0.4976],
                         std=[0.1970, 0.1970, 0.1970]),
    ])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4976, 0.4976, 0.4976],
                          std=[0.1970, 0.1970, 0.1970])
    ])

BATCH_SIZE = 64
DATA_DIR = '/Users/evgenia/OpenEyesClassificator/'

dataset = torchvision.datasets.ImageFolder(root=os.path.join(DATA_DIR, 'eyes'), transform=train_transform)

n = len(dataset)  # total number of examples
n_test = int(0.1 * n)  # take ~10% for test
test_set = torch.utils.data.Subset(dataset, range(n_test))  # take first 10%
val_set =  torch.utils.data.Subset(dataset, range(n_test, 2*n_test))  # take second 10%
train_set = torch.utils.data.Subset(dataset, range(2*n_test, n))  # take the rest   

train_loader = DataLoader(train_set, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True,  
                          num_workers=0)

val_loader = DataLoader(val_set, 
                        batch_size=BATCH_SIZE, 
                        shuffle=True, 
                        num_workers=0) 

test_loader = DataLoader(test_set, 
                         batch_size=BATCH_SIZE, 
                         shuffle=True, 
                         num_workers=0) 


# ________TRAIN_FUNCTION_______________________________________________________________________________

                    
def compute_eer(labels, scores):
    """Compute the Equal Error Rate (EER) from the predictions and scores.
    Args:
        labels (list[int]): values indicating whether the ground truth
            value is positive (1) or negative (0).
        scores (list[float]): the confidence of the prediction that the
            given sample is a positive.
    Return:
        (float, thresh): the Equal Error Rate and the corresponding threshold
    NOTES:
       The EER corresponds to the point on the ROC curve that intersects
       the line given by the equation 1 = FPR + TPR.
       The implementation of the function was taken from here:
       https://yangcha.github.io/EER-ROC/
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh

def eval(model, loader, ckpt_path=False):
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path))

    val_loss = 0.0
    total = 0
    correct = 0
    val_labels = []
    val_probs = []
    model.eval()  

    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
          
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

    return val_loss, correct / total, eer

def train(model, epoch_num, optimizer, ckpt_save_path):
    print(criterion, optimizer)
    min_val_eer = np.inf

    for epoch in range(epoch_num):
        print(epoch)
        train_loss = 0.0

        for batch in train_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print('_______')
        val_loss, val_accuracy, val_eer = eval(model, val_loader)
        test_loss, test_accuracy, test_eer = eval(model, test_loader)

        print('\033[1m' + f'Epoch {epoch}:' + '\033[0m' + f' train loss = {train_loss:.4f}')
        print(f'val loss = {val_loss:.4f}, val accuracy = {val_accuracy:.4f}, val eer = {val_eer:.4f}')
        print(f'test loss = {test_loss:.4f}, test accuracy = {test_accuracy:.4f}, test eer = {test_eer:.4f}')

        if (val_eer < min_val_eer) or (val_eer < .02):
            min_val_eer = val_eer
            torch.save(model.state_dict(), ckpt_save_path)
            print(f'Saving new weights with current val loss = {val_loss:.4f}, val accuracy = {val_accuracy:.4f}, val eer = {val_eer:.4f}, test eer = {test_eer:.4f}')


    return eer

# ________TRAIN_MODEL_______________________________________________________________________________

model = models.wide_resnet50_2(pretrained=True)
model.fc = nn.Linear(2048, 2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), 
                      lr=0.0008, 
                      momentum=0.9, 
                      nesterov=True, 
                      weight_decay=0.002)

_ = train(model, 80, optimizer, './wide_resnet50_2.pth')