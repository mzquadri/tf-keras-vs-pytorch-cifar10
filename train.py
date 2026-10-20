"""Train the Keras model on exactly the data the PyTorch baseline used.

    python train.py --out results/keras_results.json

The comparison this repository makes is only worth something if the two models
see the same images. `data/cifar10_split_indices.json` holds the exact indices
the PyTorch run selected - `torch.randperm(50000, seed=42)[:15000]`, then a
13,500/1,500 split with the same seed - exported by running torch rather than
reimplemented. Reproducing torch's RNG in NumPy is not feasible, and guessing
would have made the comparison meaningless.

Everything expressible is matched: architecture, parameter count, epochs, batch
size, optimiser, learning-rate schedule, weight decay, both dropout rates (0.25
in the conv blocks, 0.5 in the classifier head) and augmentation. What cannot be
matched is listed in README.md under "What differs, and why it cannot be
matched".
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

import model as model_module

HERE = Path(__file__).resolve().parent
SPLIT = HERE / "data" / "cifar10_split_indices.json"

# Per-channel statistics from the PyTorch transforms, used identically here.
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

EPOCHS = 12
BATCH_SIZE = 128
MAX_LR = 0.002
DROPOUT = 0.25
SEED = 42

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


class OneCycle:
    """PyTorch's OneCycleLR, as a Keras learning-rate callback.

    Keras has no OneCycle, so it is reimplemented from the PyTorch defaults
    rather than approximated with something that looks similar:

      * `pct_start=0.3` - the first 30% of steps warm up, the rest anneal
      * `div_factor=25` - initial lr is max_lr / 25
      * `final_div_factor=1e4` - final lr is initial_lr / 1e4
      * `anneal_strategy='cos'` - cosine on both the rise and the fall

    PyTorch steps the schedule per batch, not per epoch, so this does too.
    """

    def __init__(self, max_lr: float, total_steps: int, pct_start: float = 0.3,
                 div_factor: float = 25.0, final_div_factor: float = 1e4):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.warmup_steps = max(1, int(round(pct_start * total_steps)))
        self.initial_lr = max_lr / div_factor
        self.min_lr = self.initial_lr / final_div_factor

    def __call__(self, step: int) -> float:
        import math

        if step <= self.warmup_steps:
            progress = step / self.warmup_steps
            return self.initial_lr + (self.max_lr - self.initial_lr) * (
                1 - math.cos(math.pi * progress)
            ) / 2
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)
        return self.min_lr + (self.max_lr - self.min_lr) * (
            1 + math.cos(math.pi * progress)
        ) / 2


def normalise(images: np.ndarray) -> np.ndarray:
    """Scale to [0,1] then standardise per channel, as ToTensor + Normalize does."""
    x = images.astype(np.float32) / 255.0
    return (x - CIFAR10_MEAN) / CIFAR10_STD


def augment_batch(images: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """RandomCrop(32, padding=4), RandomHorizontalFlip, ColorJitter(0.2, 0.2).

    Applied to uint8 images before normalisation, which is the order the
    torchvision pipeline uses. Brightness and contrast follow torchvision's own
    definitions: brightness scales towards black, contrast scales towards the
    mean of the greyscale image.
    """
    n = images.shape[0]
    out = np.empty_like(images, dtype=np.float32)

    padded = np.pad(images, ((0, 0), (4, 4), (4, 4), (0, 0)), mode="constant")
    tops = rng.integers(0, 9, size=n)
    lefts = rng.integers(0, 9, size=n)
    flips = rng.random(n) < 0.5
    # torchvision samples each factor uniformly from [max(0, 1-b), 1+b].
    brightness = rng.uniform(0.8, 1.2, size=n).astype(np.float32)
    contrast = rng.uniform(0.8, 1.2, size=n).astype(np.float32)

    for i in range(n):
        crop = padded[i, tops[i]:tops[i] + 32, lefts[i]:lefts[i] + 32, :]
        if flips[i]:
            crop = crop[:, ::-1, :]
        img = crop.astype(np.float32)

        img = img * brightness[i]
        # Greyscale mean using the ITU-R 601-2 luma weights torchvision applies.
        grey_mean = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).mean()
        img = (img - grey_mean) * contrast[i] + grey_mean

        out[i] = np.clip(img, 0.0, 255.0)

    return out


def load_data():
    import keras

    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()

    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_idx = np.array(split["train_indices"], dtype=np.int64)
    val_idx = np.array(split["val_indices"], dtype=np.int64)

    if len(set(train_idx.tolist()) & set(val_idx.tolist())):
        raise ValueError("train and validation indices overlap")

    return (
        x_train_full[train_idx], y_train_full[train_idx].ravel(),
        x_train_full[val_idx], y_train_full[val_idx].ravel(),
        x_test, y_test.ravel(),
        split,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args(argv)

    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(SEED)

    x_tr, y_tr, x_val, y_val, x_te, y_te, split = load_data()
    print(f"train {x_tr.shape[0]}  val {x_val.shape[0]}  test {x_te.shape[0]}")

    net = model_module.build_model(dropout=DROPOUT)
    trainable = model_module.count_trainable(net)
    print(f"trainable parameters {trainable:,}")
    if trainable != model_module.EXPECTED_TRAINABLE_PARAMETERS:
        raise SystemExit(
            f"parameter count {trainable} != {model_module.EXPECTED_TRAINABLE_PARAMETERS}; "
            "the translation does not match the PyTorch model"
        )

    steps_per_epoch = int(np.ceil(x_tr.shape[0] / BATCH_SIZE))
    schedule = OneCycle(MAX_LR, total_steps=steps_per_epoch * args.epochs)

    net.compile(
        # No weight_decay argument: Keras implements it decoupled, AdamW-style,
        # while PyTorch's Adam couples it into the gradient. The equivalent is
        # the l2 regulariser in model.py, halved to account for the factor of 2
        # in the derivative of sum(w^2).
        optimizer=keras.optimizers.Adam(learning_rate=schedule.initial_lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    x_val_n = normalise(x_val)
    x_te_n = normalise(x_te)

    rng = np.random.default_rng(SEED)
    history: list[dict] = []
    best_val = 0.0
    best_epoch = 0
    global_step = 0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(x_tr.shape[0])
        running_loss, running_acc, seen = 0.0, 0.0, 0

        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            xb = normalise(augment_batch(x_tr[batch], rng))
            yb = y_tr[batch]

            global_step += 1
            net.optimizer.learning_rate.assign(schedule(global_step))

            loss, acc = net.train_on_batch(xb, yb, return_dict=False)
            running_loss += float(loss) * len(batch)
            running_acc += float(acc) * len(batch)
            seen += len(batch)

        val_loss, val_acc = net.evaluate(x_val_n, y_val, verbose=0)
        history.append({
            "epoch": epoch,
            "lr": round(float(schedule(global_step)), 8),
            "train_loss": round(running_loss / seen, 4),
            "train_accuracy": round(running_acc / seen * 100, 2),
            "val_loss": round(float(val_loss), 4),
            "val_accuracy": round(float(val_acc) * 100, 2),
        })
        print(f"epoch {epoch:>2}/{args.epochs}  "
              f"train_acc {history[-1]['train_accuracy']:.2f}%  "
              f"val_acc {history[-1]['val_accuracy']:.2f}%")

        if val_acc * 100 > best_val:
            best_val = val_acc * 100
            best_epoch = epoch

    elapsed = time.perf_counter() - started

    test_loss, test_acc = net.evaluate(x_te_n, y_te, verbose=0)
    probabilities = tf.nn.softmax(net.predict(x_te_n, verbose=0)).numpy()
    predictions = probabilities.argmax(axis=1)

    per_class = {}
    for i, name in enumerate(CLASSES):
        mask = y_te == i
        per_class[name] = round(float((predictions[mask] == i).mean()) * 100, 1)

    record = {
        "framework": "tensorflow/keras",
        "test_accuracy": round(float(test_acc) * 100, 2),
        "test_loss": round(float(test_loss), 3),
        "best_val_accuracy": round(best_val, 2),
        "best_epoch": best_epoch,
        "total_epochs": args.epochs,
        "training_time_s": round(elapsed, 1),
        "n_parameters": trainable,
        "per_class_accuracy": per_class,
        "data": {
            "train": int(x_tr.shape[0]),
            "val": int(x_val.shape[0]),
            "test": int(x_te.shape[0]),
            "split_source": split["source"],
            "split_exported_with_torch": split["torch_version"],
        },
        "hyperparameters": {
            "architecture": "Custom CNN (3 conv blocks + FC)",
            "max_epochs": args.epochs,
            "batch_size": BATCH_SIZE,
            "learning_rate": MAX_LR,
            "optimizer": "Adam",
            "scheduler": "OneCycleLR, reimplemented from the PyTorch defaults",
            "weight_decay": model_module.PYTORCH_WEIGHT_DECAY,
            "keras_l2": model_module.KERAS_L2,
            # Two rates, recorded separately: the conv blocks and the classifier
            # head differ in the reference model, so one number cannot describe both.
            "dropout_conv_blocks": DROPOUT,
            "dropout_classifier_head": model_module.HEAD_DROPOUT,
            "augmentation": [
                "RandomCrop(32, pad=4)",
                "RandomHorizontalFlip",
                "ColorJitter(brightness=0.2, contrast=0.2)",
            ],
        },
        "history": history,
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
            "machine": platform.machine(),
            "system": platform.system(),
        },
    }

    print(f"\ntest accuracy {record['test_accuracy']:.2f}%  "
          f"(PyTorch baseline 64.26%)  in {elapsed:.0f}s")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
