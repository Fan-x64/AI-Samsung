import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import FaceAutoencoder
import os, sys

IMG_SIZE = 256
MODEL_PATH = "face_autoencoder.pth"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = FaceAutoencoder(latent_channels=16).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),])

def compress(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        z = model.encode(x)
        z_uint8 = (z.cpu().numpy() * 255).astype(np.uint8)
        z_back = torch.tensor(z_uint8 / 255.0, dtype=torch.float32).to(device)
        x_hat = model.decode(z_back)

    out = (x_hat.squeeze().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(out).save("reconstructed.png")

    orig_kb = os.path.getsize(image_path) / 1024
    latent_kb = z_uint8.nbytes / 1024

    orig_arr = np.array(img.resize((IMG_SIZE, IMG_SIZE))).astype(float)
    mse = np.mean((orig_arr - out.astype(float)) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float("inf")

    tmp = "_tmp.jpg"
    img.resize((IMG_SIZE, IMG_SIZE)).save(tmp, quality=75)
    jpeg_kb = os.path.getsize(tmp) / 1024
    os.remove(tmp)

    print(f"Оригинал:    {orig_kb:.1f} KB")
    print(f"JPEG:   {jpeg_kb:.1f} KB")
    print(f"автоэнкодер:   {latent_kb:.1f} KB")
    print(f"Сжатие vs PNG:     {orig_kb/latent_kb:.1f}x")
    print(f"PSNR:              {psnr:.1f} dB")
    print(f"Сохранено:         reconstructed.png")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    compress(path)
