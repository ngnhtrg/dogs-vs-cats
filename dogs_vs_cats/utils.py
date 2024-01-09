import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm.auto import tqdm


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


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for image, label in tqdm(dataloader):
        image, label = image.to(device), label.to(device)
        # get prediction
        preds = model(image)
        loss = loss_fn(preds, label)

        # update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # add to total loss count
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, image_names, image_preds, device):
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
