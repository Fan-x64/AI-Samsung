import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import FaceAutoencoder
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

def run_demo(image_path):
    img_pil = Image.open(image_path).convert("RGB")
    x = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        z = model.encode(x)
        z_uint8 = (z.cpu().numpy() * 255).astype(np.uint8)
        z_back = torch.tensor(z_uint8 / 255.0, dtype=torch.float32).to(device)
        x_hat = model.decode(z_back)

    orig_arr = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE)))
    rec_arr = (x_hat.squeeze().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    mse = np.mean((orig_arr.astype(float) - rec_arr.astype(float)) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float("inf")

    orig_kb = os.path.getsize(image_path) / 1024
    latent_kb = z_uint8.nbytes / 1024

    tmp = "_tmp.jpg"
    img_pil.resize((IMG_SIZE, IMG_SIZE)).save(tmp, quality=75)
    jpeg_kb = os.path.getsize(tmp) / 1024
    os.remove(tmp)

    diff = np.abs(orig_arr.astype(int) - rec_arr.astype(int)).astype(np.uint8)
    diff_bright = np.clip(diff * 5, 0, 255).astype(np.uint8)

    fig = plt.figure(figsize=(12, 7))
    fig.patch.set_facecolor("#f8f8f8")
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[3, 1], hspace=0.35, wspace=0.25)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(orig_arr)
    ax0.set_title("Оригинал", fontsize=13, fontweight="bold")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(rec_arr)
    ax1.set_title(f"После автоэнкодера", fontsize=13, fontweight="bold")
    ax1.axis("off")

    ax3 = fig.add_subplot(gs[1, :])
    labels = ["Оригинал (PNG)", "JPEG", "автоэнкодер"]
    sizes = [orig_kb, jpeg_kb, latent_kb]
    colors = ["#aac4e0", "#aac4e0", "#4a90d9"]
    bars = ax3.barh(labels, sizes, color=colors, height=0.5, edgecolor="none")

    for bar, val in zip(bars, sizes):
        ax3.text(bar.get_width() + max(sizes) * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f} KB", va="center", fontsize=11)

    ax3.set_xlabel("Размер файла (KB)", fontsize=11)
    ax3.set_xlim(0, max(sizes) * 1.3)
    ax3.set_facecolor("#f8f8f8")
    ax3.spines[["top", "right", "left"]].set_visible(False)
    ax3.tick_params(left=False)

    plt.suptitle("Результат сжатия", fontsize=15, fontweight="bold")
    plt.savefig("demo_result.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"График сохранён: demo_result.png")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    run_demo(path)
