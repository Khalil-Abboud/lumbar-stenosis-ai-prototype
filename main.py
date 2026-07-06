import csv
from pathlib import Path

import torch
from torch import nn
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


# -------------------------------------------------
# 1. Paths
# -------------------------------------------------

DATA_DIR = Path("data")
LABELS_FILE = Path("labels.csv")
OUTPUT_FILE = Path("image_features.csv")


# -------------------------------------------------
# 2. Load labels
# -------------------------------------------------

labels = {}

with open(LABELS_FILE, "r", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        image_name = row["image_name"]
        label = int(row["label"])
        labels[image_name] = label


# -------------------------------------------------
# 3. Load pretrained ResNet18
# -------------------------------------------------

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Replace final classification layer with Identity.
# Now the model returns 512 image features instead of 1000 ImageNet classes.
model.fc = nn.Identity()

model.eval()

preprocess = weights.transforms()


# -------------------------------------------------
# 4. Extract features from images
# -------------------------------------------------

all_features = []

image_paths = [
    DATA_DIR / image_name
    for image_name in labels.keys()
]

for image_path in image_paths:
    image = Image.open(image_path).convert("RGB")

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        features = model(input_batch)

    feature_values = features[0].tolist()
    label = labels[image_path.name]

    all_features.append(
        [image_path.name, label] + feature_values
    )

    print()
    print("Image:", image_path.name)
    print("Label:", label)
    print("Features shape:", features.shape)
    print("First 5 features:", feature_values[:5])


# -------------------------------------------------
# 5. Save features to CSV
# -------------------------------------------------

header = ["image_name", "label"] + [
    f"feature_{i}" for i in range(512)
]

with open(OUTPUT_FILE, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(all_features)

print()
print("Done.")
print("Features saved to:", OUTPUT_FILE)
print("Number of images:", len(all_features))