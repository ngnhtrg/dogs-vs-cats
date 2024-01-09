import os

import hydra
import torch
import torch.nn as nn
import torchvision.models as models
from omegaconf import DictConfig
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import CatDogDataset, train_one_epoch


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    train_path = cfg["train"]["path_to_train"]
    train_imgs = os.listdir(train_path)
    train_imgs = [train_path + image_path for image_path in train_imgs]

    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = CatDogDataset(train_imgs, transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    pretrained_resnet = models.resnet18(weights=cfg["resnet"]["weights"])

    # Freeze all layers except the final classifier layers
    for param in pretrained_resnet.parameters():
        param.requires_grad = False

    num_cls = 1
    pretrained_resnet.fc = nn.Sequential(nn.Linear(512, num_cls), nn.Sigmoid())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pretrained_resnet.to(device)
    loss = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=cfg["resnet"]["lr"])

    NO_EPOCHS = cfg["resnet"]["no_epochs"]
    for epoch in range(NO_EPOCHS):
        ep_loss = train_one_epoch(model, train_loader, loss, optimizer, device)
        print(f"Epoch {epoch+1} loss: {ep_loss}")
    torch.save(model, cfg["train"]["model_save"])


if __name__ == "__main__":
    main()
