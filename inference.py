import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torchvision.models import Wide_ResNet50_2_Weights, ResNet34_Weights
from PIL import Image
import numpy as np
import os 


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

class OpenEyesClassifier:
    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = ResNet34_Weights.IMAGENET1K_V1.transforms()
        self.model = self.load_model()

    def load_model(self):
        model = MyNet()
        model.load_state_dict(torch.load(self.ckpt_path))
        model.to(self.device)
        model.eval()
        return model

    def predict(self, inpIm):
        img = Image.open(inpIm)
        img = img.convert('RGB')
        img = self.transform(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(self.device)

        with torch.no_grad():
            output = self.model(img)
            probs = torch.softmax(output, dim=1)
            is_open_score = probs[:, 1].cpu().numpy()
            preds = torch.argmax(probs, dim=1)

        return is_open_score
