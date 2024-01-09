import os

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from utils import evaluate


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
        return (
            self.transform(image).float() / 255.0,
            torch.Tensor([label]),
            image_path.split("/")[-1],
        )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    val_path = cfg["infer"]["path_to_val"]
    val_imgs = os.listdir(val_path)
    val_imgs = [val_path + image_path for image_path in val_imgs]

    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )

    val_dataset = CatDogDataset(val_imgs, transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_names = []
    img_preds = []

    model = torch.load(cfg["infer"]["model_path"])
    accur = evaluate(model, val_loader, img_names, img_preds, device) * 100
    val_df = pd.DataFrame({"Image Name": img_names, "Prediction": img_preds})
    val_df = val_df.reset_index(drop=True)
    val_df.to_csv(cfg["infer"]["result_save"])
    print(f"Accuracy: {accur.round()}%")


if __name__ == "__main__":
    main()
