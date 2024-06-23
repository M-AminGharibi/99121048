import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class AnimalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.images = []
        self.labels = []
        self.transform = transform
        self.label_map = {'whale': 0, 'sparrow': 1}

        for label in self.label_map.keys():
            image_dir = os.path.join(root_dir, label)
            for img_name in os.listdir(image_dir):
                self.images.append(os.path.join(image_dir, img_name))
                self.labels.append(self.label_map[label])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# تعریف تبدیلات برای تغییر اندازه تصاویر
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ایجاد دیتاست و DataLoader
root_dir = 'animal_images'
dataset = AnimalDataset(root_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
