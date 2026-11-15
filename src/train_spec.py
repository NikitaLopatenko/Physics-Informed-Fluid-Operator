"""One-shot 3D Fourier operator for periodic Navier–Stokes.

The network reads the initial velocity and, in a single pass, writes velocity
and pressure at every saved later time. Requirements from the brief are scored
as pass or fail. A miss stays a miss.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.fno3d import FNO3d
from src.solver3d import SpectralNS3d, random_velocity
from src.spectral3d import Grid3d, curl, divergence

ROOT = Path(__file__).resolve().parents[1]
NU = 0.05
T_FINAL = 0.4
DT_SOLVER = 0.02
N_SAVES = 4
REL_LIMIT = 0.03


def relative_l2(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, pred.ndim))
    num = torch.linalg.vector_norm(pred - true, dim=dims)
    den = torch.linalg.vector_norm(true, dim=dims).clamp(min=1e-8)
    return num / den


def solver_steps(n: int) -> tuple[float, int, int]:
    """Internal step small enough that RK4 is stable on the dealiased modes. Saves stay at 4."""
    k_max = (n / 3.0) * math.sqrt(3.0)
    stable = 1.2 / (NU * k_max * k_max)
    dt = min(DT_SOLVER, stable)
    n_steps = max(N_SAVES, int(math.ceil(T_FINAL / dt)))
    n_steps += (-n_steps) % N_SAVES
    return T_FINAL / n_steps, n_steps, n_steps // N_SAVES


def generate(n_samples: int, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    initial = random_velocity(n_samples, n, n_modes=3, generator=generator, dtype=torch.float64)
    dt, n_steps, save_every = solver_steps(n)
    solver = SpectralNS3d(n, NU, dt)
    with torch.no_grad():
        velocity, pressure = solver.rollout(initial, n_steps, save_every)
    return velocity.float(), pressure.float()


def future(velocity: torch.Tensor, pressure: torch.Tensor) -> torch.Tensor:
    """Pack saved times after t = 0 into channels: u, v, w, p for each time."""
    parts = []
    for t in range(1, velocity.shape[1]):
        speed = velocity[:, t]
        pres = pressure[:, t] - pressure[:, t].mean(dim=(-3, -2, -1), keepdim=True)
        parts.append(torch.cat([speed, pres.unsqueeze(1)], dim=1))
    return torch.cat(parts, dim=1)


def split_prediction(pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = pred.chunk(N_SAVES, dim=1)
    speeds = torch.stack([block[:, :3] for block in blocks], dim=1)
    pressures = torch.stack([block[:, 3] for block in blocks], dim=1)
    pressures = pressures - pressures.mean(dim=(-3, -2, -1), keepdim=True)
    return speeds, pressures


def as_velocity(raw: torch.Tensor, grid: Grid3d) -> torch.Tensor:
    """Replace each predicted potential with its curl. Pressure channels stay put.

    ∇·(∇×A) = 0, so the velocity used for the loss and the metrics is divergence-free
    without a penalty term.
    """
    blocks = []
    for block in raw.chunk(N_SAVES, dim=1):
        speed = curl(block[:, :3], grid)
        blocks.append(torch.cat([speed, block[:, 3:]], dim=1))
    return torch.cat(blocks, dim=1)


def fit(train_v, train_p, val_v, val_p, epochs: int) -> tuple[FNO3d, dict]:
    torch.manual_seed(0)
    model = FNO3d(modes=6, width=16, layers=4, out_channels=N_SAVES * 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    grid = Grid3d(train_v.shape[-1], dtype=torch.float32)
    target = future(train_v, train_p)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    best_epoch = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = as_velocity(model(train_v[:, 0]), grid)
        data = _split_rel(pred, target)
        data.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        with torch.no_grad():
            val = float(_split_rel(as_velocity(model(val_v[:, 0]), grid), future(val_v, val_p)))
        if val < best_val:
            best_val = val
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"  epoch {epoch}: train {float(data.detach()):.4f}  val {val:.4f}", flush=True)
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_val_rel_l2": best_val, "seconds": time.perf_counter() - started}


def _split_rel(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Equal weight on velocity and on pressure. One joint norm lets velocity hide the pressure error."""
    pred_speed, pred_pressure = split_prediction(pred)
    true_speed, true_pressure = split_prediction(target)
    return 0.5 * (
        relative_l2(pred_speed, true_speed).mean() + relative_l2(pred_pressure, true_pressure).mean()
    )


@torch.no_grad()
def field_errors(model: FNO3d, velocity: torch.Tensor, pressure: torch.Tensor) -> dict:
    model.eval()
    grid = Grid3d(velocity.shape[-1], dtype=torch.float32)
    pred_speed, pred_pressure = split_prediction(as_velocity(model(velocity[:, 0]), grid))
    true_speed = velocity[:, 1:]
    true_pressure = pressure[:, 1:] - pressure[:, 1:].mean(dim=(-3, -2, -1), keepdim=True)
    div = divergence(pred_speed.reshape(-1, 3, *pred_speed.shape[-3:]), grid)
    return {
        "velocity_rel_l2": float(relative_l2(pred_speed, true_speed).mean()),
        "pressure_rel_l2": float(relative_l2(pred_pressure, true_pressure).mean()),
        "divergence_rms": float(div.pow(2).mean().sqrt()),
        "divergence_max": float(div.abs().max()),
    }


def time_pair(model: FNO3d, velocity0: torch.Tensor, n: int) -> dict:
    dt, n_steps, save_every = solver_steps(n)
    solver = SpectralNS3d(n, NU, dt)
    grid = Grid3d(n, dtype=torch.float32)

    def solver_once():
        with torch.no_grad():
            solver.rollout(velocity0.double(), n_steps, save_every)

    def model_once():
        with torch.no_grad():
            as_velocity(model(velocity0.float()), grid)

    solver_once()
    model_once()
    solver_times, model_times = [], []
    for _ in range(3):
        started = time.perf_counter()
        solver_once()
        solver_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        model_once()
        model_times.append(time.perf_counter() - started)
    solver_s = float(np.median(solver_times))
    model_s = float(np.median(model_times))
    return {"solver_seconds": solver_s, "fno_seconds": model_s, "speedup_vs_this_solver": solver_s / model_s}


def try_resolution(model: FNO3d, n: int, seed: int) -> dict:
    try:
        velocity, pressure = generate(1, n, seed)
        errors = field_errors(model, velocity, pressure)
        errors["ran"] = True
        return errors
    except RuntimeError as exc:
        return {"ran": False, "error": str(exc)}


def main() -> None:
    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    print("generating 32³ trajectories", flush=True)
    train_v, train_p = generate(12, 32, seed=0)
    val_v, val_p = generate(4, 32, seed=1)
    test_v, test_p = generate(4, 32, seed=2)
    model, info = fit(train_v, train_p, val_v, val_p, epochs=40)
    torch.save(model.state_dict(), runs / "fno3d_oneshot.pt")
    test = field_errors(model, test_v, test_p)
    fine64 = try_resolution(model, 64, seed=3)
    fine128 = try_resolution(model, 128, seed=4)
    timing = time_pair(model, test_v[:1, 0], 32)
    report = {
        "setup": "One forward pass. The network predicts a vector potential and a pressure. Velocity is the curl of the potential, so divergence is a hard zero. Loss is relative L2 of velocity and pressure.",
        "nu": NU,
        "train_grid": 32,
        "A_velocity_rel_l2": test["velocity_rel_l2"],
        "A_pressure_rel_l2": test["pressure_rel_l2"],
        "A_pass_under_3pct": test["velocity_rel_l2"] < REL_LIMIT and test["pressure_rel_l2"] < REL_LIMIT,
        "B_pass_1000x_openfoam_same_gpu": False,
        "B_reason": "This machine has no NVIDIA GPU and OpenFOAM was not timed on the same GPU. The ratio below is against this spectral solver on CPU.",
        "B_timing_32": timing,
        "C_zero_shot_64": fine64,
        "C_zero_shot_128": fine128,
        "C_pass": bool(fine128.get("ran") and fine128.get("velocity_rel_l2", 1) < REL_LIMIT and fine128.get("pressure_rel_l2", 1) < REL_LIMIT),
        "D_divergence_rms": test["divergence_rms"],
        "D_divergence_max": test["divergence_max"],
        "D_pass_near_zero": test["divergence_rms"] < 1e-2,
        "train": info,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    path = runs / "spec_metrics.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in report if k != "train"}, indent=2), flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
