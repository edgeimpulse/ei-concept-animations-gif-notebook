from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def build_frame_figure(step: int, total_steps: int, output_png: Path) -> None:
    plt.close("all")
    fig, axis = plt.subplots(figsize=(5, 3), dpi=140)

    x_values = np.linspace(0, 2 * np.pi, 160)
    phase = np.interp(step, [0, max(1, total_steps - 1)], [0, 2 * np.pi])
    y_values = np.sin(x_values + phase)

    axis.plot(x_values, y_values, lw=2)
    axis.set_ylim(-1.2, 1.2)
    axis.set_title(f"DSP Template Frame {step + 1}")
    axis.set_xlabel("Time")
    axis.set_ylabel("Amplitude")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def render_dsp_animation(
    output_path: Path,
    scratch_dir: Path,
    frame_count: int = 24,
    fps: int = 8,
) -> Path:
    scratch_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []
    for step in range(frame_count):
        frame_path = scratch_dir / f"dsp_template_{step:03d}.png"
        build_frame_figure(step=step, total_steps=frame_count, output_png=frame_path)
        frame_paths.append(frame_path)

    frame_duration_ms = 1000 / max(1, fps)
    with imageio.get_writer(str(output_path), mode="I", duration=frame_duration_ms) as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))

    return output_path