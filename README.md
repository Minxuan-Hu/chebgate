# ChebGate

ChebGate is a CNN block that replaces learned spatial kernels with a **frozen depthwise operator** and a **gated Chebyshev polynomial expansion**, producing a sample-adaptive polynomial filter bank followed by **1×1 channel mixing**. The repository includes (i) ChebGate/ChebResNet training and (ii) order-mixing “program” analysis utilities.

### Environment
Tested configuration (example from our runs):
- Python 3.10.15
- PyTorch 2.5.1 + CUDA 12.1 (cuDNN 9.x)
- GPU: NVIDIA L4

### Install
Create an environment and install dependencies:
pip install -e

### Quickstart (train + benchmark with fairness harness)

DATA_DIR=$HOME/datasets
RUN_DIR=$HOME/exp/cifar10/$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR" && cd "$RUN_DIR"

PYTHONPATH=~ python3 -m chebgate.scripts.run \
  --data "$DATA_DIR" \
  --dataset cifar10 \
  --epochs 150 \
  --bs 128 \
  --stabilize_cheb 0 \
  --realization streamed \
  --amp 1 \
  --compile 1 \
  --compile_mode reduce-overhead \
  --compile_fullgraph 0 \
  --power_sample 1 \
  --power_device 0 \
  --seed 0 \
  --fair_all 1 \
  --workers 4 --prefetch 2 --persistent_workers 0 \
  --logdir "$RUN_DIR/c10_streamed_seed0" 2>&1 | tee train.log
