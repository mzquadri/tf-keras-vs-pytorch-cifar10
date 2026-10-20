"""Assert the translated model's trainable parameter count, without training it.

The README says the count is "asserted at runtime", and `train.py` does assert
it: lines 163 to 166 build the model, count the trainable weights and refuse to
start if the number is wrong. But training runs in Colab on a GPU, so on a plain
push nothing checked the claim. A README that states a verified number should
not depend on somebody remembering to train.

Building the model is the only step this needs. There is no data download, no
compile, no fit, so it costs a second on a CPU once TensorFlow is installed.

What the count is for, restated from `model.py`: two models with the same
diagram and different parameter counts are not the same model, and the count is
the cheapest way to catch a translation slip, a wrong channel width, a missing
bias, a BatchNorm on the wrong axis. It does not catch everything, and model.py
is explicit about the gap: dropout carries no weights, so a head regularised at
the conv rate would still come out at 815,018. The count is a floor, not a
proof of equivalence.

Usage:
    python check_parameters.py
"""

from __future__ import annotations

import sys

import model as model_module


def main() -> int:
    net = model_module.build_model(dropout=0.25)
    trainable = model_module.count_trainable(net)
    expected = model_module.EXPECTED_TRAINABLE_PARAMETERS

    print(f"trainable parameters {trainable:,}")
    print(f"expected             {expected:,}")

    if trainable != expected:
        print(
            f"\nFAIL: the Keras translation has {trainable:,} trainable parameters, "
            f"not {expected:,}.\n"
            "The architectures have diverged. model.py carries the per-layer "
            "arithmetic; compare it against the reference in "
            "CNN-Image-Classification-PyTorch/src/model.py.",
            file=sys.stderr,
        )
        return 1

    print("\nOK: the translation matches the PyTorch reference's parameter count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
