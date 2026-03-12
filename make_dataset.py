import os
from pathlib import Path
from torchvision.datasets import CIFAR10
from tqdm import tqdm

OUT_DIR = Path("cifar10_dataset")
CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def main():
    dataset = CIFAR10(root="raw_data", train=True, download=True)

    OUT_DIR.mkdir(exist_ok=True)

    for c in CLASSES:
        (OUT_DIR / c).mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        img, label = dataset[i]
        class_name = CLASSES[label]
        img.save(OUT_DIR / class_name / f"img_{i}.jpg")

    print("خلصنا")

if __name__ == "__main__":
    main()