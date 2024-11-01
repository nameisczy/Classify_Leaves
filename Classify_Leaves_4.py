import torch
from torch import nn, optim
from torchvision import transforms
import timm
import os

from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import time
import warnings

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
train_csv = pd.read_csv("kaggle/Classify_Leaves/train.csv")

# Label mappings
leaves_labels = sorted(list(set(train_csv['label'])))
n_classes = len(leaves_labels)
class_to_num = dict(zip(leaves_labels, range(n_classes)))
num_to_class = {v: k for k, v in class_to_num.items()}

# K-Fold Cross Validation
def kfold(data, k=5):
    KF = KFold(n_splits=k, shuffle=True, random_state=42)
    for train_idxs, test_idxs in KF.split(data):
        train_data = data.loc[train_idxs].reset_index(drop=True)
        valid_data = data.loc[test_idxs].reset_index(drop=True)
        train_iter = torch.utils.data.DataLoader(
            ReadData(train_data, train_transform), batch_size=64,
            shuffle=True, num_workers=3, pin_memory=True
        )
        valid_iter = torch.utils.data.DataLoader(
            ReadData(valid_data, valid_transform), batch_size=64,
            shuffle=True, num_workers=3, pin_memory=True
        )
        yield train_iter, valid_iter

# Dataset class
class ReadData(torch.utils.data.Dataset):
    def __init__(self, csv_data, transform=None):
        super(ReadData, self).__init__()
        self.data = csv_data
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open("kaggle/Classify_Leaves/" + self.data.loc[idx, "image"])
        label = class_to_num[self.data.loc[idx, "label"]]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)

# Data transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Data augmentation techniques
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda() if use_cuda else torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def color(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda() if use_cuda else torch.randperm(batch_size)
    new_x = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)(x)
    y_a, y_b = y, y[index]
    return new_x, y_a, y_b, lam

def flip_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda() if use_cuda else torch.randperm(batch_size)
    new_x = transforms.RandomRotation(degrees=(90, 180))(x)
    y_a, y_b = y, y[index]
    return new_x, y_a, y_b, lam

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1, bby1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
    bbx2, bby2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda() if use_cuda else torch.randperm(batch_size)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

# Mixup criterion
def mixup_criterion(pred, y_a, y_b, lam):
    return lam * nn.CrossEntropyLoss()(pred, y_a) + (1 - lam) * nn.CrossEntropyLoss()(pred, y_b)

criterion = mixup_criterion

# Load models with modifications
def get_models(k=5):
    models = {}
    for mk in range(k):
        model = timm.create_model("resnest50d_4s2x40d", False, drop_rate=.5)
        in_features = model.get_classifier().in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(.3),
            nn.Linear(512, len(num_to_class))
        )

        model.load_state_dict(torch.load(f"best_model_fold0.pth"))
        for i, param in enumerate(model.children()):
            if i == 6:
                break
            param.requires_grad = False

        model.to(device)

        opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 10, T_mult=2)
        models[f"model_{mk}"] = {
            "model": model,
            "opt": opt,
            "scheduler": scheduler,
            "last_acc": .98
        }

    return models

models = get_models()

# Training function
def train_model():
    best_valid_acc = 0  # Initialize the best validation accuracy
    for epoch in range(10):
        fold_train_acc = []
        fold_valid_acc = []
        for k, (train_iter, valid_iter) in enumerate(kfold(train_csv, 5)):
            model = models[f"model_{k}"]["model"]
            opt = models[f"model_{k}"]["opt"]
            scheduler = models[f"model_{k}"]["scheduler"]
            s = time.time()
            model.train()
            train_loss = []
            train_acc = 0
            length = 0
            for x, y in train_iter:
                x, y = x.to(device), y.to(device)
                random_num = np.random.random()
                if random_num <= 1 / 4:
                    x, y_a, y_b, lam = mixup_data(x, y, use_cuda=True)
                elif random_num <= 1 / 2:
                    x, y_a, y_b, lam = cutmix_data(x, y, use_cuda=True)
                elif random_num <= 3 / 4:
                    x, y_a, y_b, lam = flip_data(x, y, use_cuda=True)
                else:
                    x, y_a, y_b, lam = mixup_data(x, y, alpha=0, use_cuda=True)
                x, y_a, y_b = map(torch.autograd.Variable, (x, y_a, y_b))
                output = model(x)
                loss = criterion(output, y_a, y_b, lam)
                train_loss.append(loss.item())
                predict = output.argmax(dim=1)
                length += x.shape[0]
                train_acc += lam * (predict == y_a).cpu().sum().item() + \
                             (1 - lam) * (predict == y_b).cpu().sum().item()
                opt.zero_grad()
                loss.backward()
                opt.step()
                scheduler.step()

            model.eval()
            valid_acc = []
            with torch.no_grad():
                for x, y in valid_iter:
                    x, y = x.to(device), y.to(device)
                    pre_x = model(x)
                    valid_acc.append((pre_x.argmax(1) == y).float().mean().item())

            k_train_ = train_acc / length
            k_valid_ = sum(valid_acc) / len(valid_acc)
            fold_train_acc.append(k_train_)
            fold_valid_acc.append(k_valid_)

            # Save the model after each epoch
            torch.save(model.state_dict(), f"Resnest50d_fold{k}_epoch{epoch + 1}.pth")
            print(f"Model saved: Resnest50d_fold{k}_epoch{epoch + 1}.pth")

            response = f"Epoch {epoch + 1}-Fold{k + 1} —— " + \
                       f"Train Loss: {sum(train_loss) / len(train_loss) :.3f}, " + \
                       f"Train Accuracy: {k_train_ * 100 :.2f}%, " + \
                       f"Valid Accuracy: {k_valid_ * 100 :.2f}%, " + \
                       f"Learning Rate: {opt.param_groups[0]['lr'] :.6f}, " + \
                       f"Time Out: {time.time() - s :.1f}s"
            print(response)

        t_accuracy = np.mean(fold_train_acc)
        v_accuracy = np.mean(fold_valid_acc)
        print(f"Epoch {epoch + 1} —— " + \
              f"Train Accuracy: {t_accuracy * 100 :.2f}%, " + \
              f"Valid Accuracy: {v_accuracy * 100 :.2f}%\n")

        # Check if the current epoch's validation accuracy is the best
        if v_accuracy > best_valid_acc:
            best_valid_acc = v_accuracy
            for k in range(5):
                torch.save(models[f"model_{k}"]["model"].state_dict(), f"best_model_fold{k}.pth")
                print(f"Best model saved for fold {k}: best_model_fold{k}.pth")

class TestData(torch.utils.data.Dataset):
    def __init__(self, csv_data, transform=None):
        super(TestData, self).__init__()
        self.data = csv_data
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open("kaggle/Classify_Leaves/" + self.data.loc[idx, "image"])
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    train_model()

    test_csv = pd.read_csv("kaggle/Classify_Leaves/test.csv")

    test_iter = torch.utils.data.DataLoader(
        TestData(test_csv, valid_transform), batch_size=128,
        num_workers=2, pin_memory=True
    )

    predict = None
    with torch.no_grad():
        for x in test_iter:
            x = x.to(device)
            p = torch.zeros((x.size()[0], len(class_to_num))).to(device)
            for k in range(5):
                model = models[f"model_{k}"]["model"]
                p += model(x).detach()
            if predict is None:
                predict = p.argmax(1)
            else:
                predict = torch.cat([predict, p.argmax(1)])

    df = pd.read_csv("kaggle/Classify_Leaves/sample_submission.csv")
    df.label = predict.cpu().numpy()
    df.label = df.label.apply(lambda x: num_to_class[x])
    df.to_csv("kaggle/Classify_Leaves/submission.csv", index=False)