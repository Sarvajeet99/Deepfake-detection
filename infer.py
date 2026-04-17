# infer.py
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import torch
from PIL import Image
from model import CLASS_NAMES, get_eval_transform, get_model

device = "cuda" if torch.cuda.is_available() else "cpu"

model = get_model(pretrained=False)
checkpoint = torch.load("model.pth", map_location=device)
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
    class_names = checkpoint.get("class_names", CLASS_NAMES)
else:
    state_dict = checkpoint
    class_names = CLASS_NAMES

model.load_state_dict(state_dict)
model.to(device)
model.eval()

transform = get_eval_transform()

parser = argparse.ArgumentParser(description="Run deepfake inference on an image.")
parser.add_argument(
    "image_path",
    nargs="*",
    help="Path to the image file to test, including .jpg, .jpeg, or .png. If omitted, a file picker opens.",
)
args = parser.parse_args()

def choose_image_path():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select an image for inference",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return path


def resolve_image_path(path_parts):
    if not path_parts:
        return None

    raw_path = " ".join(path_parts).strip().strip('"').strip("'")
    candidate = Path(raw_path).expanduser()
    candidates = [candidate]

    if not candidate.is_absolute():
        candidates.append(Path.cwd() / candidate)
        candidates.append(Path.home() / "Downloads" / candidate.name)

    for option in candidates:
        if option.exists() and option.is_file():
            return option.resolve()

    return None


image_path = resolve_image_path(args.image_path)
if image_path is None:
    image_path = choose_image_path()

if not image_path:
    raise SystemExit("No image selected.")

image_path = Path(image_path).expanduser().resolve()
img = Image.open(image_path).convert("RGB")
img = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(img)
    probs = torch.softmax(logits, dim=1)[0]
    predicted_idx = torch.argmax(probs).item()

real_pct = probs[0].item() * 100
fake_pct = probs[1].item() * 100
predicted_label = class_names[predicted_idx]

print(f"{real_pct:.1f}% real")
print(f"{fake_pct:.1f}% fake")
print(f"Image: {image_path}")
print(f"Prediction: {predicted_label}")
