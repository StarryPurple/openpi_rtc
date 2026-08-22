"""Synthetic smoke test for ``eval_offline_rtc.evaluate_file`` (no openpi).

Builds a fake HDF5 in the XTrainer schema (qpos + 3 compressed image streams +
action), a fake policy that mimics the JAX ``Policy.infer`` surface
(``infer`` / ``last_raw_chunk`` / ``prepare_prev_chunk``), then runs
``evaluate_file`` in baseline and RTC modes and asserts the data flow and
metrics.

Run:  uv run python openpi_rtc/tests/test_eval_offline_smoke.py
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import h5py
import numpy as np

from openpi_rtc import eval_offline_rtc as eo

T = 60
DIM = 14
CHUNK = 20


def make_fake_hdf5(path: str, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    traj = np.cumsum(rng.normal(0, 0.05, (T, DIM)), axis=0)
    with h5py.File(path, "w") as f:
        f.create_dataset("/observations/qpos", data=traj.astype(np.float32))
        f.create_dataset("/action", data=traj.astype(np.float32))
        for cam in ["top", "left_wrist", "right_wrist"]:
            frames = []
            for i in range(T):
                img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
                ok, buf = cv2.imencode(".jpg", img)
                assert ok
                frames.append(buf.tobytes())
            n = max(len(b) for b in frames)
            dt = f.create_dataset(f"/observations/images/{cam}", (T, n),
                                  dtype="uint8", chunks=(1, n))
            for i, b in enumerate(frames):
                arr = np.frombuffer(b, dtype="uint8")
                dt[i, : len(arr)] = arr


class FakePolicy:
    """Mimics the JAX Policy.infer surface used by evaluate_file."""

    def __init__(self, rtc: bool):
        self.rtc = rtc
        self.calls = []
        self._raw = np.zeros((CHUNK, DIM), dtype=np.float32)

    def infer(self, obs, **kwargs):
        self.calls.append((sorted(obs.keys()), dict(kwargs)))
        if self.rtc:
            self._raw = np.random.randn(CHUNK, DIM).astype(np.float32)
        return {"actions": np.random.randn(CHUNK, DIM).astype(np.float32)}

    @property
    def last_raw_chunk(self):
        return self._raw if self.rtc else None

    def prepare_prev_chunk(self, prev_raw, prev_state, cur_state):
        return np.asarray(prev_raw)


def make_args(tmp: Path, mode: str, stride: int) -> argparse.Namespace:
    return argparse.Namespace(
        mode=mode,
        config="pi05-task_00031_yulong-xtrainer",
        checkpoint="fake/49999",
        dataset=str(tmp / "episode_0000.hdf5"),
        prompt="Transfer the test tube from the right rack to the left rack.",
        max_steps=None,
        log_dir=str(tmp / "logs"),
        inference_delay=4,
        execution_horizon=10,
        max_guidance_weight=10.0,
        schedule="exp",
        stride=stride,
        anchor_correction=True,
    )


def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        h5 = tmp / "episode_0000.hdf5"
        make_fake_hdf5(str(h5))

        # ---- baseline mode (stride 1) ----
        pol = FakePolicy(rtc=False)
        st_b = eo.evaluate_file(str(h5), pol, make_args(tmp, "baseline", 1))
        assert st_b is not None
        for k in ("mse", "l1", "steps", "mean_infer_ms"):
            assert k in st_b, f"baseline missing {k}"
        assert st_b["steps"] > 0
        assert all(kw == {} for _, kw in pol.calls), "baseline infer must have no rtc kwargs"
        print(f"baseline OK: mse={st_b['mse']:.4f} steps={st_b['steps']} "
              f"infer_ms={st_b['mean_infer_ms']:.1f}")

        # ---- rtc mode (stride == d) ----
        pol2 = FakePolicy(rtc=True)
        st_r = eo.evaluate_file(str(h5), pol2, make_args(tmp, "rtc", 4))
        assert st_r is not None
        for k in ("mse", "l1", "steps", "mean_infer_ms", "boundary_mse", "boundary_l1"):
            assert k in st_r, f"rtc missing {k}"
        rtc_calls = [kw for _, kw in pol2.calls]
        assert rtc_calls[0] == {}, "first rtc call must be plain (no prev chunk)"
        assert all(
            "prev_chunk_left_over" in kw and "inference_delay" in kw
            for kw in rtc_calls[1:]
        ), "rtc infer missing guidance kwargs"
        print(f"rtc OK: mse={st_r['mse']:.4f} boundary_mse={st_r['boundary_mse']:.4f} "
              f"steps={st_r['steps']}")

    print("smoke test passed")


if __name__ == "__main__":
    main()
