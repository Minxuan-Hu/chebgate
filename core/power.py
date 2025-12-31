import time
import threading
import subprocess
from typing import List
import numpy as np


class GPUPowerSampler:
    """
    Samples GPU power (Watts) periodically in a background thread.
    Also (optionally) collects util% and used-MB.
    Prefers NVML (pynvml). Falls back to `nvidia-smi`. If neither works, becomes a no-op.
    Provides robust stats() when samples are missing.
    """
    def __init__(self, interval: float = 0.2, device_index: int = 0, method: str = "auto"):
        import shutil as _shutil
        self.interval = float(interval)
        self.device_index = int(device_index)
        self.method = method
        self._run = False
        self._thr = None
        self._samples_t: List[float] = []
        self._samples_w: List[float] = []
        self._samples_util_g: List[float] = []
        self._samples_util_m: List[float] = []
        self._samples_mem_mb: List[float] = []
        self._missing = 0
        self._detected_method = "none"
        self._nvml = None
        self._nvml_handle = None
        self._smi_ok = False

        # Try NVML
        if method in ("auto", "nvml"):
            try:
                import pynvml  # type: ignore
                pynvml.nvmlInit()
                h = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
                _ = pynvml.nvmlDeviceGetPowerUsage(h)  # probe
                self._nvml = pynvml
                self._nvml_handle = h
                self._detected_method = "nvml"
            except Exception:
                self._nvml = None
                self._nvml_handle = None

        # Try nvidia-smi
        if self._detected_method == "none" and method in ("auto", "smi"):
            self._smi_ok = _shutil.which("nvidia-smi") is not None
            if self._smi_ok:
                try:
                    _ = self._read_smi()  # probe
                    self._detected_method = "smi"
                except Exception:
                    self._smi_ok = False

    def _read_nvml(self):
        try:
            mw = self._nvml.nvmlDeviceGetPowerUsage(self._nvml_handle)  # milliwatts
            w = float(mw) / 1000.0
            util = self._nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            u_g = float(util.gpu) if util is not None else None
            u_m = float(util.memory) if util is not None else None
            mem = self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            mem_mb = float(mem.used) / (1024 * 1024) if mem is not None else None
            return w, u_g, u_m, mem_mb
        except Exception:
            return None, None, None, None

    def _read_smi(self):
        try:
            out = subprocess.check_output([
                "nvidia-smi",
                "--query-gpu=power.draw,utilization.gpu,utilization.memory,memory.used",
                "--format=csv,noheader,nounits",
                "-i", str(self.device_index)
            ])
            s = out.decode("utf-8").strip().splitlines()[0]
            parts = [p.strip() for p in s.split(",")]
            w = float(parts[0]) if parts[0] else None
            u_g = float(parts[1]) if len(parts) > 1 and parts[1] else None
            u_m = float(parts[2]) if len(parts) > 2 and parts[2] else None
            mem_mb = float(parts[3]) if len(parts) > 3 and parts[3] else None
            return w, u_g, u_m, mem_mb
        except Exception:
            return None, None, None, None

    def _poll_once(self):
        t = time.perf_counter()
        if self._detected_method == "nvml":
            w, u_g, u_m, mem_mb = self._read_nvml()
        elif self._detected_method == "smi":
            w, u_g, u_m, mem_mb = self._read_smi()
        else:
            w = u_g = u_m = mem_mb = None
        if w is None:
            self._missing += 1
        self._samples_t.append(t); self._samples_w.append(w)
        self._samples_util_g.append(u_g); self._samples_util_m.append(u_m)
        self._samples_mem_mb.append(mem_mb)

    def _loop(self):
        self._poll_once()
        while self._run:
            time.sleep(self.interval)
            self._poll_once()

    def start(self):
        if self._detected_method == "none":
            return
        self._run = True
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def stop(self):
        if self._thr is None:
            return
        self._run = False
        self._thr.join(timeout=5.0)
        # Clean NVML to avoid warnings in long multi-run sessions
        if self._detected_method == "nvml" and self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    def _nanmean(self, xs: List[float]):
        vs = [x for x in xs if x is not None]
        return float(np.nanmean(vs)) if vs else None

    def _nanmax(self, xs: List[float]):
        vs = [x for x in xs if x is not None]
        return float(np.nanmax(vs)) if vs else None

    def _nanmin(self, xs: List[float]):
        vs = [x for x in xs if x is not None]
        return float(np.nanmin(vs)) if vs else None

    def stats(self) -> dict:
        ts = self._samples_t
        ws = self._samples_w
        n = len(ts)
        dur = (ts[-1] - ts[0]) if n >= 2 else 0.0

        # Integrate energy only over adjacent valid samples
        energy_j = 0.0
        for i in range(1, n):
            w0, w1 = ws[i - 1], ws[i]
            if (w0 is None) or (w1 is None):
                continue
            dt = ts[i] - ts[i - 1]
            energy_j += 0.5 * (w0 + w1) * dt

        return {
            "method": self._detected_method,
            "interval_s": self.interval,
            "device_index": self.device_index,
            "samples": n,
            "missing": int(self._missing),
            "duration_s": float(dur),
            "mean_watts": self._nanmean(ws),
            "min_watts": self._nanmin(ws),
            "max_watts": self._nanmax(ws),
            "mean_util_gpu": self._nanmean(self._samples_util_g),
            "mean_util_mem": self._nanmean(self._samples_util_m),
            "mean_mem_used_mb": self._nanmean(self._samples_mem_mb),
            "energy_joules": float(energy_j),
        }
