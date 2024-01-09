import os

import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm


# Path to directory with image dataset and subfolders for training, validation
DATA_PATH = os.path.dirname(os.getcwd()) + "/data"
MODEL_SAVE_PATH = os.path.dirname(os.getcwd()) + "/models"

TRAIN_PATH = DATA_PATH + "/train_11k/"
train_imgs = os.listdir(TRAIN_PATH)
train_imgs = [TRAIN_PATH + image_path for image_path in train_imgs]


class CatDogDataset(Dataset):
    def __init__(self, folder_contents, transform):
        self.folder_contents = folder_contents
        self.transform = transform

    def __len__(self):
        return len(self.folder_contents)

    def __getitem__(self, idx):
        image_path = self.folder_contents[idx]
        image = Image.open(image_path)
        label = 0  # 0 is dog, 1 is cat
        if "cat." in image_path:
            label = 1
        return self.transform(image).float() / 255.0, torch.Tensor([label])


transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)

train_dataset = CatDogDataset(train_imgs, transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


# Load a pre-trained ResNet18 model (you can choose other models as well)
pretrained_resnet = models.resnet18(weights="DEFAULT")

# Freeze all layers except the final classifier layers
for param in pretrained_resnet.parameters():
    param.requires_grad = False

# Modify the classifier layers for your specific task
num_classes = 1  # Replace with the number of classes in your dataset
pretrained_resnet.fc = nn.Sequential(nn.Linear(512, num_classes), nn.Sigmoid())


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for image, label in tqdm(dataloader):
        image, label = image.to(device), label.to(device)
        # get prediction
        preds = model(image)
        loss = loss_fn(preds, label)

        # update weights
        optimizer.zero_grad()  # Clear the gradients
        loss.backward()  # Compute gradients
        optimizer.step()

        # add to total loss count
        total_loss += loss.item()
    return total_loss / len(dataloader)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = pretrained_resnet.to(device)
loss = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=1e-4)

NO_EPOCHS = 1
for epoch in range(NO_EPOCHS):
    epoch_loss = train_one_epoch(model, train_loader, loss, optimizer, device)
    print(f"Epoch {epoch+1} loss: {epoch_loss}")
torch.save(model, MODEL_SAVE_PATH + "/resnet.pth")
