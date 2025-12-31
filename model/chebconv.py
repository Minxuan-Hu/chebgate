import torch
import torch.nn as nn
import torch.nn.functional as F

from chebgate.core.fp32 import fp32_reference_mode


class ChebConv2d(nn.Module):
    """
    Chebyshev convolution layer with:
      - Fixed depthwise 3×3 Laplacian (λ)
      - Trainable order scales α_k and an order gate g_k(x)
      - Four exact realizations (same weights/params/MACs math):
        (A) 'streamed' : accumulate per-order 1×1 during recurrence
        (B) 'concat'   : concatenate [s_k·U_k] then a single 1×1
        (C) 'gemm'     : OBC via reshape → GEMM → reshape (identical math to einsum contraction)
        (D) 'mstream'  : micro-streamed — uniform K-loop variant of 'streamed'
    All compute y = ∑_k s_k · (U_k * W_k), with s_k = α_k · g_k(x).
    """

    def __init__(
        self,
        Cin,
        Cout,
        K,
        lap=0.25,
        r=8,
        realization="streamed",
        gate_mode="on",
        stabilize_cheb: bool = True,
    ):
        super().__init__()
        assert realization in ("streamed", "concat", "gemm", "mstream")
        assert gate_mode in ("on", "off")

        self.Cin, self.Cout, self.K = Cin, Cout, K
        self.realization = realization
        self.gate_mode = gate_mode
        self.lap_lambda = float(lap)

        # Scale A = a_scale * L so ||A|| ≤ 1 when enabled
        self.a_scale = min(1.0, 1.0 / (8.0 * self.lap_lambda)) if stabilize_cheb else 1.0

        base = torch.tensor(
            [[0.0, lap, 0.0], [lap, -4 * lap, lap], [0.0, lap, 0.0]],
            dtype=torch.float32,
        )
        ker = base.unsqueeze(0).unsqueeze(0).expand(Cin, 1, 3, 3).clone()
        self.register_buffer("base_L", ker, persistent=False)

        self.lap = nn.Conv2d(Cin, Cin, 3, padding=1, groups=Cin, bias=False)
        with torch.no_grad():
            self.lap.weight.copy_(self.base_L)
            self.lap.weight.requires_grad = False

        # α_k (learned order scales)
        self.order_scales = nn.Parameter(torch.ones(K + 1) * lap)

        # gate g_k(x)
        hidden = max(Cin // r, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(Cin, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, K + 1, 1, bias=True),
            nn.Sigmoid(),
        )

        # Shared 1×1 weights (packed as [Cout, Cin*(K+1), 1, 1])
        self.combine = nn.Conv2d(Cin * (K + 1), Cout, 1, bias=False)

        # gate stats buffers (eval only)
        self.register_buffer("gate_sum", torch.zeros(K + 1))
        self.register_buffer("gate_sqsum", torch.zeros(K + 1))
        self.register_buffer("gate_count", torch.tensor(0, dtype=torch.long))

    def _gate_tensor(self, x):
        if self.gate_mode == "off":
            return torch.ones((x.size(0), self.K + 1, 1, 1), device=x.device, dtype=x.dtype)
        return self.gate(self.pool(x))

    def _accum_gate_stats(self, g):
        if self.training:
            return
        with torch.no_grad():
            g2 = g.reshape(g.size(0), g.size(1))
            self.gate_sum += g2.sum(dim=0).to(self.gate_sum.dtype)
            self.gate_sqsum += (g2 * g2).sum(dim=0).to(self.gate_sqsum.dtype)
            self.gate_count += g.size(0)

    def reset_gate_stats(self):
        self.gate_sum.zero_()
        self.gate_sqsum.zero_()
        self.gate_count.zero_()

    def gate_stats(self):
        n = int(self.gate_count.item())
        if n == 0:
            return None
        mean = (self.gate_sum / n).cpu().tolist()
        var = (self.gate_sqsum / n - (self.gate_sum / n) ** 2).cpu().tolist()
        std = [float(max(v, 0.0) ** 0.5) for v in var]
        return {
            "mean": mean,
            "std": std,
            "count": n,
            "lambda_lap": self.lap_lambda,
            "a_scale": float(self.a_scale),
        }

    def _build_U_list(self, x):
        U0 = x
        if self.K == 0:
            return [U0]
        U1 = self.a_scale * self.lap(x)
        U_list = [U0, U1]
        for _k in range(2, self.K + 1):
            LU = self.a_scale * self.lap(U_list[-1])
            U_new = 2.0 * LU - U_list[-2]
            U_list.append(U_new)
        return U_list

    @torch.no_grad()
    def exactness_check(self, x: torch.Tensor) -> dict:
        """
        Strict FP32 reference equivalence check across realizations:
          - autocast disabled
          - TF32 disabled temporarily (matmul + cuDNN)
          - reports max/mean absolute differences

        Note: Different realizations may not be bitwise identical due to different
        reduction/accumulation orders; we report numeric deltas instead.
        """
        was_training = bool(self.training)
        device_type = "cuda" if x.is_cuda else "cpu"

        try:
            self.eval()
            with fp32_reference_mode(strict_tf32_off=True, disable_cudnn_benchmark=True):
                with torch.autocast(device_type=device_type, enabled=False):
                    x32 = x.detach().to(dtype=torch.float32)
                    y_stream = self._forward_streamed(x32, record_gate=False)
                    y_concat = self._forward_concat(x32, record_gate=False)
                    y_mstream = self._forward_mstream(x32, record_gate=False)
                    y_gemm = self._forward_gemm(x32, record_gate=False)
        finally:
            if was_training:
                self.train()

        def diffs(a, b):
            d = (a - b).abs()
            return {"max_abs": float(d.max().item()), "mean_abs": float(d.mean().item())}

        return {
            "mode": {
                "autocast": "disabled",
                "tf32": "disabled",
                "dtype": "fp32",
                "note": "deltas reflect floating-point reassociation / kernel differences across realizations",
            },
            "streamed_vs_concat": diffs(y_stream, y_concat),
            "streamed_vs_mstream": diffs(y_stream, y_mstream),
            "streamed_vs_gemm": diffs(y_stream, y_gemm),
            "concat_vs_mstream": diffs(y_concat, y_mstream),
            "concat_vs_gemm": diffs(y_concat, y_gemm),
            "mstream_vs_gemm": diffs(y_mstream, y_gemm),
        }

    def _forward_streamed(self, x, record_gate=True):
        g = self._gate_tensor(x)
        if record_gate:
            self._accum_gate_stats(g)

        comb_w = self.combine.weight
        W_chunks = torch.split(comb_w, self.Cin, dim=1)

        s_all = self.order_scales.to(x.dtype).view(1, -1, 1, 1) * g

        y = F.conv2d(x, W_chunks[0], bias=None, stride=1, padding=0) * s_all[:, 0:1]
        if self.K == 0:
            return y.contiguous(memory_format=torch.channels_last)

        U_prev2 = x
        U_prev1 = self.a_scale * self.lap(x)
        y = y + F.conv2d(U_prev1, W_chunks[1], bias=None, stride=1, padding=0) * s_all[:, 1:2]

        for k in range(2, self.K + 1):
            LU = self.a_scale * self.lap(U_prev1)
            U_new = 2.0 * LU - U_prev2
            y = y + F.conv2d(U_new, W_chunks[k], bias=None, stride=1, padding=0) * s_all[:, k : k + 1]
            U_prev2, U_prev1 = U_prev1, U_new

        return y.contiguous(memory_format=torch.channels_last)

    def _forward_concat(self, x, record_gate=True):
        g = self._gate_tensor(x)
        if record_gate:
            self._accum_gate_stats(g)

        s = self.order_scales.to(x.dtype).view(1, -1, 1, 1) * g
        U_list = self._build_U_list(x)

        V = torch.cat([U_list[k] * s[:, k : k + 1] for k in range(self.K + 1)], dim=1)
        V = V.contiguous(memory_format=torch.channels_last)
        y = self.combine(V)
        return y.contiguous(memory_format=torch.channels_last)

    def _forward_gemm(self, x, record_gate=True):
        """
        GEMM realization:
          - forms Ub = stack_k(U_k) with gating,
          - flattens to (NHW, K1*Cin),
          - multiplies by shared packed weights,
          - reshapes back to (N, Cout, H, W).

        Identical math to contracting over {order, Cin}.
        """
        g = self._gate_tensor(x)
        if record_gate:
            self._accum_gate_stats(g)

        s = self.order_scales.to(x.dtype).view(1, -1, 1, 1) * g  # [N,K+1,1,1]
        U_list = self._build_U_list(x)  # list of [N,Cin,H,W]
        Ub = torch.stack(U_list, dim=1)  # [N,K+1,Cin,H,W]
        Ub = Ub * s.view(Ub.size(0), Ub.size(1), 1, 1, 1)

        N, K1, Cin, H, W = Ub.shape

        # A: (NHW, K1*Cin)
        A = Ub.permute(0, 3, 4, 1, 2).reshape(N * H * W, K1 * Cin).contiguous()
        # B: (K1*Cin, Cout) from the same 1×1 combine weights
        B = self.combine.weight.reshape(self.Cout, K1 * Cin).transpose(0, 1).contiguous()
        # GEMM
        Y = A @ B  # (NHW, Cout)

        y = Y.view(N, H, W, self.Cout).permute(0, 3, 1, 2)
        return y.contiguous(memory_format=torch.channels_last)

    def _forward_mstream(self, x, record_gate=True):
        """
        Micro-streamed realization:
          - same Chebyshev recurrence and 1×1s as 'streamed',
          - no materialization of (K+1)*Cin feature stack,
          - expressed as a single uniform K loop.
        """
        g = self._gate_tensor(x)
        if record_gate:
            self._accum_gate_stats(g)

        s_all = self.order_scales.to(x.dtype).view(1, -1, 1, 1) * g  # [N,K+1,1,1]
        N, Cin, H, W = x.shape
        y = torch.zeros(N, self.Cout, H, W, device=x.device, dtype=x.dtype)

        # View combine weights as W_k: [Cout, K+1, Cin, 1, 1]
        W_all = self.combine.weight.view(self.Cout, self.K + 1, self.Cin, 1, 1)

        U_prev2 = x  # U_0
        U_prev1 = None
        if self.K >= 1:
            U_prev1 = self.a_scale * self.lap(x)  # U_1

        for k in range(self.K + 1):
            if k == 0:
                Uk = U_prev2
            elif k == 1:
                Uk = U_prev1
            else:
                LU = self.a_scale * self.lap(U_prev1)
                Uk = 2.0 * LU - U_prev2
                U_prev2, U_prev1 = U_prev1, Uk

            w_k = W_all[:, k, :, :, :]  # [Cout,Cin,1,1]
            y_k = F.conv2d(Uk, w_k, bias=None, stride=1, padding=0)
            sk = s_all[:, k : k + 1]  # [N,1,1,1]
            y = y + y_k * sk

        return y.contiguous(memory_format=torch.channels_last)

    def forward(self, x):
        if self.realization == "streamed":
            return self._forward_streamed(x, record_gate=True)
        if self.realization == "concat":
            return self._forward_concat(x, record_gate=True)
        if self.realization == "gemm":
            return self._forward_gemm(x, record_gate=True)
        # 'mstream'
        return self._forward_mstream(x, record_gate=True)
