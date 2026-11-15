"""Train a one-step FNO against the spectral solver and record its own speed."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.fno import FNO2d
from src.solver import SpectralNS, random_vorticity
from src.spectral import SpectralGrid, divergence, vorticity_residual

ROOT = Path(__file__).resolve().parents[1]
NU = 1e-2
DT_SAVE = 0.1
MODES = 12
WIDTH = 32
LAYERS = 4


def relative_l2(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, pred.ndim))
    num = torch.linalg.vector_norm(pred - true, dim=dims)
    den = torch.linalg.vector_norm(true, dim=dims).clamp(min=1e-8)
    return num / den


def make_trajectories(
    n_samples: int,
    n: int,
    nu: float,
    dt: float,
    t_final: float,
    n_modes: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    omega0 = random_vorticity(n_samples, n, n_modes, generator, torch.float64)
    solver = SpectralNS(n, nu, dt, dtype=torch.float64)
    n_steps = int(round(t_final / dt))
    save_every = int(round(DT_SAVE / dt))
    with torch.no_grad():
        frames = solver.rollout(omega0, n_steps, save_every)
    return frames.float()


def pairs(frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    now = frames[:, :-1].reshape(-1, 1, frames.shape[-2], frames.shape[-1])
    nxt = frames[:, 1:].reshape(-1, 1, frames.shape[-2], frames.shape[-1])
    return now, nxt


def physics_term(pred: torch.Tensor, now: torch.Tensor, true_next: torch.Tensor, nu: float, grid: SpectralGrid) -> torch.Tensor:
    residual = vorticity_residual(now.squeeze(1), pred.squeeze(1), DT_SAVE, nu, grid)
    with torch.no_grad():
        scale = (true_next.squeeze(1) - now.squeeze(1)).pow(2).mean() / DT_SAVE**2
    return residual.pow(2).mean() / scale.clamp(min=1e-8)


def epoch_pass(
    model: FNO2d,
    now: torch.Tensor,
    nxt: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    nu: float,
    grid: SpectralGrid,
    physics_weight: float,
    batch: int,
) -> float:
    train = optimizer is not None
    model.train(train)
    order = torch.randperm(now.shape[0]) if train else torch.arange(now.shape[0])
    total = 0.0
    count = 0
    for start in range(0, now.shape[0], batch):
        index = order[start : start + batch]
        batch_now = now[index]
        batch_next = nxt[index]
        if train:
            optimizer.zero_grad(set_to_none=True)
        pred = model(batch_now)
        data = relative_l2(pred, batch_next).mean()
        loss = data
        if physics_weight > 0:
            loss = loss + physics_weight * physics_term(pred, batch_now, batch_next, nu, grid)
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += float(data.detach()) * len(index)
        count += len(index)
    return total / max(count, 1)


@torch.no_grad()
def one_step_error(model: FNO2d, frames: torch.Tensor) -> float:
    model.eval()
    now, nxt = pairs(frames)
    pred = model(now)
    return float(relative_l2(pred, nxt).mean())


@torch.no_grad()
def rollout_error(model: FNO2d, frames: torch.Tensor) -> float:
    model.eval()
    current = frames[:, 0]
    pred_final = current
    for _ in range(frames.shape[1] - 1):
        pred_final = model(current.unsqueeze(1)).squeeze(1)
        current = pred_final
    return float(relative_l2(pred_final, frames[:, -1]).mean())


@torch.no_grad()
def persistence_error(frames: torch.Tensor) -> float:
    now, nxt = pairs(frames)
    return float(relative_l2(now, nxt).mean())


def fit(name: str, frames_train: torch.Tensor, frames_val: torch.Tensor, physics_weight: float, epochs: int, lr: float, batch: int, nu: float) -> tuple[FNO2d, dict]:
    torch.manual_seed(0)
    model = FNO2d(MODES, WIDTH, LAYERS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    grid = SpectralGrid(frames_train.shape[-1], dtype=torch.float32)
    now, nxt = pairs(frames_train)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_val = math.inf
    best_epoch = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss = epoch_pass(model, now, nxt, optimizer, nu, grid, physics_weight, batch)
        val = one_step_error(model, frames_val)
        history.append({"epoch": epoch, "train_rel_l2": train_loss, "val_rel_l2": val})
        if val < best_val:
            best_val = val
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        print(f"  {name} epoch {epoch}: train {train_loss:.4f}  val {val:.4f}", flush=True)
    model.load_state_dict(best_state)
    return model, {
        "physics_weight": physics_weight,
        "best_epoch": best_epoch,
        "best_val_rel_l2": best_val,
        "seconds": time.perf_counter() - started,
        "parameters": sum(p.numel() for p in model.parameters()),
        "history": history,
    }


@torch.no_grad()
def residual_stats(model: FNO2d, frames: torch.Tensor, nu: float) -> dict:
    model.eval()
    grid = SpectralGrid(frames.shape[-1], dtype=torch.float32)
    now, nxt = pairs(frames)
    pred = model(now)
    residual = vorticity_residual(now.squeeze(1), pred.squeeze(1), DT_SAVE, nu, grid)
    true_residual = vorticity_residual(now.squeeze(1), nxt.squeeze(1), DT_SAVE, nu, grid)
    div = divergence(pred.squeeze(1), grid)
    return {
        "prediction_residual_rms": float(residual.pow(2).mean().sqrt()),
        "solver_step_residual_rms": float(true_residual.pow(2).mean().sqrt()),
        "prediction_divergence_max": float(div.abs().max()),
    }


def time_rollout(model: FNO2d, solver: SpectralNS, omega0: torch.Tensor, n_steps: int, save_every: int) -> dict:
    def solver_once():
        with torch.no_grad():
            solver.rollout(omega0.double(), n_steps, save_every)

    def model_once():
        with torch.no_grad():
            current = omega0.float()
            jumps = n_steps // save_every
            for _ in range(jumps):
                current = model(current.unsqueeze(1)).squeeze(1)

    solver_once()
    model_once()
    solver_times = []
    model_times = []
    for _ in range(5):
        started = time.perf_counter()
        solver_once()
        solver_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        model_once()
        model_times.append(time.perf_counter() - started)
    solver_s = float(np.median(solver_times))
    model_s = float(np.median(model_times))
    return {
        "solver_seconds": solver_s,
        "fno_seconds": model_s,
        "snapshots": n_steps // save_every,
        "speedup": solver_s / model_s if model_s > 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the 2D vorticity FNO")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train-samples", type=int, default=48)
    parser.add_argument("--val-samples", type=int, default=8)
    parser.add_argument("--test-samples", type=int, default=8)
    parser.add_argument("--fine-samples", type=int, default=4)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--nu", type=float, default=NU)
    parser.add_argument("--t-final", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--physics-weight", type=float, default=0.1)
    args = parser.parse_args()

    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    dt = 0.01
    print("generating 32² trajectories", flush=True)
    train = make_trajectories(args.train_samples, args.n, args.nu, dt, args.t_final, n_modes=4, seed=0)
    val = make_trajectories(args.val_samples, args.n, args.nu, dt, args.t_final, n_modes=4, seed=1)
    test = make_trajectories(args.test_samples, args.n, args.nu, dt, args.t_final, n_modes=4, seed=2)
    baseline = persistence_error(test)
    print(f"persistence one-step rel L2 on test: {baseline:.4f}", flush=True)

    data_model, data_info = fit("data", train, val, 0.0, args.epochs, args.lr, args.batch, args.nu)
    phys_model, phys_info = fit("physics", train, val, args.physics_weight, args.epochs, args.lr, args.batch, args.nu)
    torch.save(data_model.state_dict(), runs / "fno_data.pt")
    torch.save(phys_model.state_dict(), runs / "fno_physics.pt")

    def evaluate(model: FNO2d) -> dict:
        return {
            "test_one_step_rel_l2": one_step_error(model, test),
            "test_rollout_final_rel_l2": rollout_error(model, test),
            **residual_stats(model, test, args.nu),
        }

    print("generating 64² trajectories for the resolution check", flush=True)
    fine_dt = 0.005
    fine = make_trajectories(args.fine_samples, 64, args.nu, fine_dt, args.t_final, n_modes=4, seed=3)
    report = {
        "equation": "2D periodic vorticity Navier-Stokes, no forcing",
        "nu": args.nu,
        "saved_step": DT_SAVE,
        "grid_train": args.n,
        "split": "separate initial conditions, seeds 0/1/2",
        "persistence_one_step_rel_l2": baseline,
        "data_only": {**data_info, "test": evaluate(data_model)},
        "physics": {**phys_info, "test": evaluate(phys_model)},
        "zero_shot_64": {
            "data_only_one_step_rel_l2": one_step_error(data_model, fine),
            "data_only_rollout_final_rel_l2": rollout_error(data_model, fine),
            "physics_one_step_rel_l2": one_step_error(phys_model, fine),
            "physics_rollout_final_rel_l2": rollout_error(phys_model, fine),
            "persistence_one_step_rel_l2": persistence_error(fine),
        },
        "timing_32": time_rollout(data_model, SpectralNS(32, args.nu, dt), test[:1, 0].double(), int(args.t_final / dt), int(DT_SAVE / dt)),
        "timing_64": time_rollout(data_model, SpectralNS(64, args.nu, fine_dt), fine[:1, 0].double(), int(args.t_final / fine_dt), int(DT_SAVE / fine_dt)),
        "caveat": "Speedup is this CPU, this grid, and this error. It is not the 256² figure from Li et al.",
    }
    # Histories make the file long but they are the training record. Keep them.
    path = runs / "metrics.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    _print_report(report)
    print(f"wrote {path}", flush=True)


def _print_report(report: dict) -> None:
    print(f"persistence {report['persistence_one_step_rel_l2']:.4f}", flush=True)
    for name in ("data_only", "physics"):
        test = report[name]["test"]
        print(
            f"{name}: one-step {test['test_one_step_rel_l2']:.4f}"
            f"  rollout {test['test_rollout_final_rel_l2']:.4f}"
            f"  residual {test['prediction_residual_rms']:.4f}"
            f"  (solver step {test['solver_step_residual_rms']:.4f})"
            f"  div {test['prediction_divergence_max']:.2e}",
            flush=True,
        )
    fine = report["zero_shot_64"]
    print(
        f"64² zero-shot one-step data {fine['data_only_one_step_rel_l2']:.4f}"
        f"  physics {fine['physics_one_step_rel_l2']:.4f}"
        f"  persistence {fine['persistence_one_step_rel_l2']:.4f}",
        flush=True,
    )
    for label in ("timing_32", "timing_64"):
        row = report[label]
        print(
            f"{label}: solver {row['solver_seconds']:.4f}s  fno {row['fno_seconds']:.4f}s  speedup {row['speedup']:.2f}×",
            flush=True,
        )


if __name__ == "__main__":
    main()
