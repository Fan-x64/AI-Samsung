import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from model import FaceAutoencoder
import os

IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 30
DATA_DIR = "./data/faces"
MODEL_PATH = "face_autoencoder.pth"

class FaceDataset(Dataset):
    def __init__(self, folder, transform):
        self.paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.transform = transform
        print(f"Найдено изображений: {len(self.paths)}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)

transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),])

dataset = FaceDataset(DATA_DIR, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Используется: GPU")
else:
    device = torch.device("cpu")
    print("Используется: CPU")

model = FaceAutoencoder(latent_channels=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(EPOCHS):
    total_loss = 0
    for imgs in loader:
        imgs = imgs.to(device)
        reconstructed = model(imgs)
        loss = loss_fn(reconstructed, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Эпоха {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), MODEL_PATH)
print(f"Модель сохранена: {MODEL_PATH}")
