
from __future__ import annotations

import logging
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

logger = logging.getLogger(__name__)

LOW_RAM_GB = 2.5
FLOOR_RAM_GB = 1.0
MIN_BATCH = 8


def mps_smoke_test() -> bool:
    import torch

    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return False
    try:
        torch.manual_seed(0)
        src = torch.randn(6, 4)
        idx = torch.tensor([0, 0, 1, 1, 2, 2])
        gru = torch.nn.GRUCell(4, 4)
        lstm = torch.nn.LSTMCell(4, 4)
        A = torch.randn(6, 4, 4)

        def run(dev: str):
            s, i = src.to(dev), idx.to(dev)
            out = torch.zeros(3, 4, device=dev).index_add_(0, i, s)
            mx = torch.full((3,), float("-inf"), device=dev)
            mx.scatter_reduce_(0, i, s[:, 0], reduce="amax", include_self=False)
            g = gru.to(dev)(s, torch.zeros(6, 4, device=dev))
            h, c = lstm.to(dev)(s, (torch.zeros(6, 4, device=dev), torch.zeros(6, 4, device=dev)))
            b = torch.bmm(A.to(dev), s.unsqueeze(-1).to(dev)).squeeze(-1)
            return [t.detach().cpu() for t in (out, mx, g, h, c, b)]

        cpu, mps = run("cpu"), run("mps")
        return all(torch.allclose(a, b, atol=1e-4) for a, b in zip(cpu, mps))
    except Exception as exc:
        logger.warning("MPS smoke test failed (%s); using CPU", exc)
        return False


def pick_device(prefer: str | None = None) -> str:
    prefer = prefer or os.environ.get("TOXION_DEVICE")
    if prefer:
        return prefer
    return "mps" if mps_smoke_test() else "cpu"


def available_ram_gb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.virtual_memory().available / 1e9


def estimate_edge_memory_mb(batch_size: int, hidden_dim: int, edges_per_molecule: int = 54) -> float:
    return batch_size * edges_per_molecule * hidden_dim * hidden_dim * 4 / 1e6


def choose_batch_size(requested: int, available_gb: float | None = None) -> int:
    avail = available_ram_gb() if available_gb is None else available_gb
    if avail is None:
        return requested
    if avail < FLOOR_RAM_GB:
        raise MemoryError(f"only {avail:.1f} GB RAM free; close other applications before training")
    if avail < LOW_RAM_GB:
        reduced = max(MIN_BATCH, requested // 2)
        logger.warning("%.1f GB RAM free: batch size %d -> %d", avail, requested, reduced)
        return reduced
    return requested
