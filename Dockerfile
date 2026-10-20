# CPU-only TensorFlow, pinned, for a reproducible training run.
#
# tensorflow-cpu rather than tensorflow: this trains on a CPU and the default
# wheel carries the CUDA libraries, which are several gigabytes that would never
# be loaded. The same trap the insureassist image had with torch.
FROM python:3.11-slim

RUN pip install --no-cache-dir \
      "tensorflow-cpu==2.20.0" \
      "numpy==2.1.3"

WORKDIR /work
