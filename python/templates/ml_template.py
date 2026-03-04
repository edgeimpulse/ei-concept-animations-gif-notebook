from __future__ import annotations

from pathlib import Path

import gif
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


@gif.frame
def build_frame(step: int, total_steps: int) -> None:
    plt.close("all")
    fig, axis = plt.subplots(figsize=(5, 3), dpi=140)

    x_values = np.linspace(-8, 8, 220)
    shift = np.interp(step, [0, max(1, total_steps - 1)], [-4, 4])
    y_values = 1 / (1 + np.exp(-(x_values - shift)))

    axis.plot(x_values, y_values, lw=2)
    axis.set_ylim(-0.05, 1.05)
    axis.set_title(f"ML Template Frame {step + 1}")
    axis.set_xlabel("Input")
    axis.set_ylabel("Output")
    axis.grid(alpha=0.3)
    fig.tight_layout()


def render_ml_animation(
    output_path: Path,
    frame_count: int = 24,
    hold_last: int = 6,
    duration_ms: int = 100,
) -> Path:
    frames = [build_frame(step, frame_count) for step in range(frame_count)]
    frames.extend([build_frame(frame_count - 1, frame_count) for _ in range(max(0, hold_last))])
    gif.save(frames, str(output_path), duration=duration_ms)
    return output_path