"""Train LeNet-5 on upright MNIST and on randomly rotated MNIST, then plot confusion matrices."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def make_loaders(data_dir: str, batch_size: int, rotate: bool, seed: int):
    normalize = transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    train_tf = [transforms.ToTensor(), normalize]
    if rotate:
        train_tf.insert(0, transforms.RandomRotation(180))
    train_set = datasets.MNIST(
        data_dir, train=True, download=True, transform=transforms.Compose(train_tf)
    )

    # Test set is materialized once so the rotated variant uses fixed angles.
    test_raw = datasets.MNIST(data_dir, train=False, download=True)
    images = test_raw.data.unsqueeze(1).float() / 255.0
    if rotate:
        gen = torch.Generator().manual_seed(seed)
        angles = torch.empty(len(images)).uniform_(-180, 180, generator=gen)
        images = torch.stack(
            [TF.rotate(img, float(a)) for img, a in zip(images, angles)]
        )
    images = (images - MNIST_MEAN) / MNIST_STD
    test_set = TensorDataset(images, test_raw.targets)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(test_set, batch_size=1024)
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, device, desc):
    model.train()
    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        y_pred.append(model(x.to(device)).argmax(1).cpu())
        y_true.append(y)
    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()


def run(name, rotate, args, device):
    torch.manual_seed(args.seed)
    train_loader, test_loader = make_loaders(
        args.data_dir, args.batch_size, rotate, args.seed
    )
    model = LeNet5().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        train_epoch(model, train_loader, optimizer, device, f"{name} epoch {epoch}")
        y_true, y_pred = evaluate(model, test_loader, device)
        print(f"[{name}] epoch {epoch}: test acc {(y_true == y_pred).mean():.4f}")
    return y_true, y_pred


def plot_confusion(results, out):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5.5))
    for ax, (title, (y_true, y_pred)) in zip(axes, results.items()):
        cm = confusion_matrix(y_true, y_pred, labels=range(10), normalize="true")
        acc = (y_true == y_pred).mean()
        ax.imshow(np.pow(cm, 1 / 3), cmap="Greens", vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                if cm[i, j] > 0.001:
                    ax.text(
                        j,
                        i,
                        f"{100 * cm[i, j]:.1f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white" if cm[i, j] > 0.5 else "black",
                    )
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        # ax.set_title(f"{title}\n(acc {100 * acc:.2f}%)")
    fig.suptitle("61k CNN")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved figure to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="confusion_matrices.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = {
        "Trained & tested on MNIST": run("upright", False, args, device),
        "Trained & tested on rotated MNIST": run("rotated", True, args, device),
    }
    for title, (y_true, y_pred) in results.items():
        print(f"{title}: acc {np.mean(y_true == y_pred):.4f}")
    plot_confusion(results, args.out)


if __name__ == "__main__":
    main()
