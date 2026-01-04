# ChebGate

ChebGate is a CNN block that replaces learned spatial kernels with a **frozen depthwise operator** and a **gated Chebyshev polynomial expansion**, producing a sample-adaptive polynomial filter bank followed by **1×1 channel mixing**. The repository includes (i) ChebGate/ChebResNet training and (ii) order-mixing “program” analysis utilities.

Paper summary and reported results are in the accompanying SPL manuscript. :contentReference[oaicite:0]{index=0}

## Environment
Tested configuration (example from our runs):
- Python 3.10.15
- PyTorch 2.5.1 + CUDA 12.1 (cuDNN 9.x)
- GPU: NVIDIA L4
