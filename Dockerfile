# CPU-only TensorFlow, pinned, for a reproducible training run.
#
# tensorflow-cpu rather than tensorflow: this trains on a CPU and the default
# wheel carries the CUDA libraries, which are several gigabytes that would never
# be loaded. The same trap the insureassist image had with torch.
#
# The versions live in requirements.txt rather than here, because CI installs
# the same two packages to check the parameter count and a pin written twice is
# a pin that drifts.
FROM python:3.11-slim

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /work
