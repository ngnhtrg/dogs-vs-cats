import os
import zipfile
import torch
from torchvision import transforms
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from PIL import Image
import pandas as pd
import numpy as np

DATA_PATH = os.path.dirname(os.getcwd()) + '/data'
MODEL_SAVE_PATH = os.path.dirname(os.getcwd()) + '/models'
RESULT_PATH = os.path.dirname(os.getcwd()) + '/results'
PATH_TO_VAL_ZIP_FILE = DATA_PATH + '/val.zip'
with zipfile.ZipFile(PATH_TO_VAL_ZIP_FILE, 'r') as zip_ref:
    zip_ref.extractall(DATA_PATH)
VAL_PATH = DATA_PATH + '/val/'
val_imgs = os.listdir(VAL_PATH)
val_imgs = [VAL_PATH + image_path for image_path in val_imgs]

class CatDogDataset(Dataset):
    def __init__(self, folder_contents, transform):
        self.folder_contents = folder_contents
        self.transform = transform
        
    def __len__(self):
        return len(self.folder_contents)
    
    def __getitem__(self, idx):
        image_path = self.folder_contents[idx]
        image = Image.open(image_path)
        label = 0 # 0 is dog, 1 is cat
        if "cat" in image_path: label = 1
        return self.transform(image).float() / 255.0, torch.Tensor([label]), image_path.split('/')[-1]

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

val_dataset = CatDogDataset(val_imgs, transform)
val_loader = DataLoader(val_dataset, batch_size=16)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

image_names = []
image_preds = []

def evaluate(model, dataloader):
    model.eval()
    total_correct = 0
    total = 0
    for image, label, img_name in tqdm(dataloader):
        image, label = image.to(device), label.to(device)
        # get prediction
        preds = model(image).round()
        correct = (preds == label).sum()
        total += preds.shape[0]
        total_correct += correct
        image_names.extend(list(img_name))
        image_preds.extend(preds.detach().numpy().astype(int))
    return total_correct / total

model = torch.load(MODEL_SAVE_PATH + '/resnet.pth')
accuracy = evaluate(model, val_loader) * 100
val_df = pd.DataFrame({'Image Name': image_names, 'Prediction': image_preds})
val_df = val_df.reset_index(drop=True)
val_df.to_csv(RESULT_PATH + '/out.csv')
print(f"Accuracy: {accuracy.round()}%")