"""Train the TruthLens real-vs-AI image classifier.

Dataset : Hemg/AI-Generated-vs-Real-Images-Datasets (Hugging Face)
Output  : models/truthlens_best.pth   (loaded automatically by model.py / main.py)
          models/metrics.json         (validation accuracy per epoch)
          samples/                    (held-out images for a guaranteed-good demo)

Typical use on Google Colab (free T4 GPU, ~10 minutes):
    pip install -r requirements-train.txt
    python train.py --samples 12000 --epochs 3

Quick pipeline check with no internet and no dataset (CPU is fine):
    python train.py --synthetic --samples 64 --epochs 1 --no-pretrained --batch-size 16
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from arch import CLASSES, INPUT_SIZE, MEAN, STD, TruthLensClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
SAMPLES_DIR = os.path.join(HERE, "samples")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "truthlens_best.pth")

_resize = transforms.Resize(INPUT_SIZE)  # identical to the resize used at inference


# --------------------------------------------------------------------------- data
def resolve_label_map(names, ai_index_override):
    """Map the dataset's label ids to our ids: 0 = authentic, 1 = ai_generated."""
    if ai_index_override is not None:
        return {i: (1 if i == ai_index_override else 0) for i in range(len(names))}

    mapping = {}
    for i, name in enumerate(names):
        n = name.lower()
        if any(k in n for k in ("real", "authentic", "human", "natural")):
            mapping[i] = 0
        elif any(k in n for k in ("ai", "fake", "gen", "synth")):
            mapping[i] = 1
        else:
            raise SystemExit(
                f"Cannot tell whether label '{name}' is real or AI. "
                f"Labels are {names}; re-run with --ai-label-index <id>."
            )
    if set(mapping.values()) != {0, 1}:
        raise SystemExit(
            f"Label names {names} did not resolve to one real and one AI class; "
            "re-run with --ai-label-index <id>."
        )
    return mapping


def to_array(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    return np.asarray(_resize(img), dtype=np.uint8)


def load_huggingface(args):
    from datasets import load_dataset

    print(f"Loading {args.dataset} (first run downloads the dataset)...")
    ds = load_dataset(args.dataset, split=args.split)
    print(f"Dataset has {len(ds):,} rows; columns: {ds.column_names}")

    names = ds.features[args.label_col].names
    label_map = resolve_label_map(names, args.ai_label_index)
    print("Label mapping (dataset label -> TruthLens class):")
    for i, n in enumerate(names):
        print(f"  {i} '{n}' -> {CLASSES[label_map[i]]}")

    ds = ds.shuffle(seed=args.seed).select(range(min(args.samples, len(ds))))
    xs, ys, skipped = [], [], 0
    t0 = time.time()
    for k, row in enumerate(ds):
        try:
            xs.append(to_array(row[args.image_col]))
            ys.append(label_map[row[args.label_col]])
        except Exception:
            skipped += 1
        if (k + 1) % 1000 == 0:
            print(f"  decoded {k + 1}/{len(ds)} images ({time.time() - t0:.0f}s)")
    if skipped:
        print(f"Skipped {skipped} unreadable images")
    return np.stack(xs), np.array(ys, dtype=np.int64)


def make_synthetic(n, seed):
    """Tiny stand-in dataset so the pipeline can be tested offline."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for i in range(n):
        label = i % 2
        if label == 0:  # smooth gradient + gentle noise
            g = np.linspace(0, 255, 224, dtype=np.float32)
            base = np.stack([np.tile(g, (224, 1)), np.tile(g[::-1], (224, 1)), np.full((224, 224), 120, np.float32)], axis=2)
            img = base + rng.normal(0, 6, base.shape)
        else:  # high-frequency checker pattern
            c = ((np.indices((224, 224)).sum(0) // 4) % 2) * 255
            img = np.stack([c, 255 - c, c], axis=2).astype(np.float32) + rng.normal(0, 6, (224, 224, 3))
        xs.append(np.clip(img, 0, 255).astype(np.uint8))
        ys.append(label)
    return np.stack(xs), np.array(ys, dtype=np.int64)


class ArrayDataset(Dataset):
    def __init__(self, x, y, train):
        self.x, self.y, self.train = x, y, train
        self.mean = torch.tensor(MEAN).view(3, 1, 1)
        self.std = torch.tensor(STD).view(3, 1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        t = torch.from_numpy(self.x[i]).permute(2, 0, 1).float() / 255.0
        if self.train:
            if random.random() < 0.5:
                t = torch.flip(t, dims=[2])  # horizontal flip
            t = (t * (0.9 + 0.2 * random.random())).clamp(0, 1)  # brightness jitter
        return (t - self.mean) / self.std, int(self.y[i])


# ------------------------------------------------------------------------ training
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    per_class = {0: [0, 0], 1: [0, 0]}  # class -> [correct, total]
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        total += len(yb)
        for c in (0, 1):
            m = yb == c
            per_class[c][0] += (pred[m] == c).sum().item()
            per_class[c][1] += m.sum().item()
    recall = {CLASSES[c]: (v[0] / v[1] if v[1] else 0.0) for c, v in per_class.items()}
    return correct / max(total, 1), recall


def save_samples(x_val, y_val, per_class=6):
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    counts = {0: 0, 1: 0}
    for img, y in zip(x_val, y_val):
        if counts[int(y)] >= per_class:
            continue
        name = "real" if y == 0 else "ai"
        Image.fromarray(img).save(os.path.join(SAMPLES_DIR, f"{name}_{counts[int(y)] + 1}.png"))
        counts[int(y)] += 1
        if all(v >= per_class for v in counts.values()):
            break


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="Hemg/AI-Generated-vs-Real-Images-Datasets")
    p.add_argument("--split", default="train")
    p.add_argument("--image-col", default="image")
    p.add_argument("--label-col", default="label")
    p.add_argument("--ai-label-index", type=int, default=None, help="Force which dataset label id means AI-generated")
    p.add_argument("--samples", type=int, default=12000, help="Images to use in total (train + validation)")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true", help="Do not download ImageNet weights")
    p.add_argument("--synthetic", action="store_true", help="Use a tiny generated dataset (pipeline test only)")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu" and not args.synthetic:
        print("WARNING: no GPU found. ResNet50 on CPU is very slow; use Colab (Runtime > Change runtime type > T4 GPU).")

    x, y = make_synthetic(args.samples, args.seed) if args.synthetic else load_huggingface(args)
    print(f"Loaded {len(y):,} images  (real={int((y == 0).sum())}, ai={int((y == 1).sum())})")

    idx = np.random.permutation(len(y))
    n_val = max(2, int(len(y) * args.val_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    x_tr, y_tr, x_val, y_val = x[train_idx], y[train_idx], x[val_idx], y[val_idx]

    pin = device.type == "cuda"
    train_loader = DataLoader(ArrayDataset(x_tr, y_tr, True), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin, drop_last=len(y_tr) > args.batch_size)
    val_loader = DataLoader(ArrayDataset(x_val, y_val, False), batch_size=args.batch_size,
                            num_workers=args.workers, pin_memory=pin)

    model = TruthLensClassifier(num_classes=2, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    os.makedirs(MODELS_DIR, exist_ok=True)
    history, best_acc = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for step, (xb, yb) in enumerate(train_loader, 1):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * len(yb)
            seen += len(yb)
            if step % 20 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {running / seen:.4f}")

        val_acc, recall = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1), "val_acc": val_acc, "recall": recall})
        print(f"Epoch {epoch}/{args.epochs}  loss {running / max(seen, 1):.4f}  val_acc {val_acc:.4f}  "
              f"recall {recall}  ({time.time() - t0:.0f}s)")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "meta": {"classes": CLASSES, "val_acc": val_acc, "epoch": epoch, "dataset": args.dataset,
                         "synthetic": args.synthetic},
            }, WEIGHTS_PATH)
            print(f"  saved best model -> {WEIGHTS_PATH}")

    # Prove the saved file loads into the exact inference architecture.
    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu")
    TruthLensClassifier(num_classes=2).load_state_dict(ckpt["state_dict"], strict=True)
    print("Checkpoint verified: loads into the inference model with strict=True")

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump({"best_val_acc": best_acc, "history": history, "train_images": len(y_tr),
                   "val_images": len(y_val), "synthetic": args.synthetic}, f, indent=2)
    if not args.synthetic:
        save_samples(x_val, y_val)
        print(f"Saved demo images to {SAMPLES_DIR}")
    print(f"Done. Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
