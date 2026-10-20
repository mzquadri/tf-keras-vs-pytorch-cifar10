# TensorFlow/Keras vs PyTorch on CIFAR-10

A translation, not a race. The PyTorch model in
[CNN-Image-Classification-PyTorch](https://github.com/mzquadri/CNN-Image-Classification-PyTorch)
reached 64.26% on CIFAR-10 from 15,000 training images in twelve epochs. This repository
rebuilds that model in Keras and trains it on **exactly the same images**, to find out how
much of the gap between two frameworks is the framework and how much is everything else.

The interesting part is not the accuracy number. It is the list of things that *cannot* be
matched, and what each one costs.

## Project Overview

| Component | Details |
|-----------|---------|
| **Dataset** | CIFAR-10, downloaded by `keras.datasets.cifar10.load_data()` |
| **Training subset** | 13,500 train + 1,500 validation = 15,000 images |
| **Test set** | 10,000 images (the full CIFAR-10 test split) |
| **Architecture** | Custom 3-block CNN with Batch Normalization |
| **Parameters** | **815,018 trainable**, asserted at runtime |
| **Framework** | TensorFlow / Keras |
| **PyTorch baseline** | 64.26% test accuracy |
| **Keras test accuracy** | *pending the Colab run - see `results/`* |

## The split is exported, not reimplemented

A framework comparison is worth nothing if the two models see different images.

`data/cifar10_split_indices.json` holds the exact indices the PyTorch run selected:
`torch.randperm(50000, generator=manual_seed(42))[:15000]`, then
`random_split([13500, 1500], manual_seed(42))`. Those indices were **exported by running
torch**, not reproduced in NumPy. Torch's Mersenne Twister stream cannot be replicated
with `numpy.random`, and guessing would have quietly made the comparison meaningless.

```json
{
  "source": "torch.randperm(50000, generator=manual_seed(42))[:15000], then random_split([13500,1500], manual_seed(42))",
  "torch_version": "2.14.1+cpu",
  "subset_size": 15000
}
```

CIFAR-10 itself is never committed. Keras downloads it, and the index file is the only
thing that travels.

## Architecture

```
Input (32x32x3)
    |
Conv Block 1: [Conv3x3(3->32) -> BN -> ReLU] x2 -> MaxPool -> SpatialDropout2D(0.25)
    |   Output: 16x16x32
Conv Block 2: [Conv3x3(32->64) -> BN -> ReLU] x2 -> MaxPool -> SpatialDropout2D(0.25)
    |   Output: 8x8x64
Conv Block 3: [Conv3x3(64->128) -> BN -> ReLU] x2 -> MaxPool -> SpatialDropout2D(0.25)
    |   Output: 4x4x128
Flatten -> Dense(2048->256) -> ReLU -> Dropout(0.5) -> Dense(256->10)
    |
Output (10 logits)
```

### The parameter count is asserted, not trusted

```
conv1a  3*32*9   + 32  =    896      bn1a  2*32  =   64
conv1b  32*32*9  + 32  =  9,248      bn1b  2*32  =   64
conv2a  32*64*9  + 64  = 18,496      bn2a  2*64  =  128
conv2b  64*64*9  + 64  = 36,928      bn2b  2*64  =  128
conv3a  64*128*9 + 128 = 73,856      bn3a  2*128 =  256
conv3b  128*128*9+ 128 =147,584      bn3b  2*128 =  256
dense1  2048*256 + 256 =524,544
dense2  256*10   + 10  =  2,570
                          -------
                          815,018
```

`train.py` exits non-zero if the built model disagrees with that number. Two models with
the same block diagram and different parameter counts are not the same model, and the
count is the cheapest way to catch a wrong channel width, a missing bias, or a
BatchNorm on the wrong axis. BatchNorm contributes `2*C` trainable (gamma, beta) and
`2*C` non-trainable (running mean, variance); PyTorch's count includes only the
trainable ones, so that is what is compared.

## What differs, and why it cannot be matched

| Difference | Why it is unavoidable | How it was handled |
|---|---|---|
| **Weight decay** | PyTorch's `Adam` adds `wd * w` to the *gradient*. Keras's `l2` adds `l * sum(w^2)` to the *loss*, whose gradient is `2 * l * w`. | `l = wd / 2 = 5e-5`. Using `1e-4` directly would silently double the regularisation. |
| **OneCycleLR** | Keras has no OneCycle. | Reimplemented from the PyTorch defaults (`pct_start=0.3`, `div_factor=25`, `final_div_factor=1e4`, cosine both ways), stepped **per batch** as PyTorch does. |
| **Channel dropout** | `nn.Dropout2d` drops whole channels; `keras.layers.Dropout` drops individual activations. Same rate, different regulariser. | `SpatialDropout2D` in the conv blocks. |
| **Augmentation RNG** | Torch and NumPy cannot be made to draw the same random crops and flips. | The torchvision *definitions* are reimplemented in NumPy (ITU-R 601-2 luma weights for contrast, brightness before contrast), so the distribution matches even though the draws do not. |
| **Kernel nondeterminism** | cuDNN/BLAS reduction orders differ between frameworks and between devices. | Accepted. This is the irreducible floor on any such comparison. |

### Two dropout rates, which is easy to get wrong

The reference model does not use one dropout rate, it uses two: `nn.Dropout2d(0.25)` in
the conv blocks and `nn.Dropout(0.5)` in the classifier head. Reusing the conv rate for
the head halves the regularisation there, and because dropout carries no weights the
parameter count still lands on 815,018 - so the assertion sails straight past it. The
head rate therefore lives in its own constant, `model.HEAD_DROPOUT`, and both rates are
recorded separately in the result JSON.

## Running it

Locally, with the pinned CPU image:

```bash
docker build -t tf-cifar .
docker run --rm -v "$PWD":/work tf-cifar python train.py --out results/keras_results.json
```

Or on Colab, which is how the tracked result was produced:

```
notebooks/colab_keras_cifar10.ipynb
```

Upload `train.py`, `model.py` and `data/cifar10_split_indices.json`; the notebook checks
the split and the parameter count before training, then trains and re-checks the number
inside the saved JSON.

## Repository layout

```
train.py                              the training run; asserts the parameter count
model.py                              the Keras translation and the hand-worked count
Dockerfile                            pinned tensorflow-cpu 2.20.0 / numpy 2.1.3
data/cifar10_split_indices.json       the 15,000 indices, exported from torch
notebooks/colab_keras_cifar10.ipynb   the Colab run
results/                              training records (JSON)
```

## Data source

CIFAR-10, fetched at runtime by `keras.datasets.cifar10.load_data()`. Collected by Alex
Krizhevsky, Vinod Nair and Geoffrey Hinton
([dataset page](https://www.cs.toronto.edu/~kriz/cifar.html)). No image data is
redistributed in this repository.

## Built with AI assistance (Claude Code)

The Keras translation, the parameter-count derivation, the OneCycle reimplementation and
the Colab notebook were written with AI assistance (Claude Code). The numeric claims are
asserted in code rather than stated in prose, so they can be checked by running them.

## Licence

MIT - see [LICENSE](LICENSE).
