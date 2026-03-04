from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import gif
import imageio.v2 as imageio
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Ellipse, Rectangle

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_GAPMINDER_URL = (
    "https://github.com/aeturrell/coding-for-economists/raw/main/data/owid_gapminder.csv"
)
DEFAULT_STYLE_URL = (
    "https://github.com/aeturrell/coding-for-economists/raw/main/plot_style.txt"
)

# Global playback control: 2.0 means animations run at half speed.
SLOWDOWN_FACTOR = 2.0

PRESET_NAMES: tuple[str, ...] = (
    "dsp_sine_shift",
    "dsp_processing_blocks",
    "dsp_processing_blocks_individual",
    "nn_sigmoid_shift",
    "ml_learning_blocks",
    "ml_learning_blocks_individual",
    "sine_bead",
    "gapminder_full",
)

DEFAULT_ALL_PRESETS: tuple[str, ...] = (
    "dsp_sine_shift",
    "dsp_processing_blocks",
    "nn_sigmoid_shift",
    "ml_learning_blocks",
    "sine_bead",
    "gapminder_full",
)

DSP_PROCESSING_BLOCKS: tuple[str, ...] = (
    "Raw Data",
    "Flatten",
    "Image",
    "Spectral features",
    "Spectrogram",
    "Audio MFE",
    "Audio MFCC",
    "Audio Syntiant",
    "IMU Syntiant",
    "HR/HRV features",
)

ML_LEARNING_BLOCKS: tuple[str, ...] = (
    "Classification (Keras)",
    "Regression (Keras)",
    "Anomaly Detection (K-means)",
    "Anomaly Detection (GMM)",
    "Visual anomaly detection (FOMO-AD)",
    "Image Classification (Transfer Learning)",
    "Keyword Spotting (Transfer Learning)",
    "Object Detection (MobileNetV2 SSD FPN)",
    "Object Detection (FOMO)",
    "Classical ML",
    "Custom block (PyTorch/Keras/scikit-learn)",
)


try:
    from pygifsicle import optimize as _optimize_gif
except Exception:
    _optimize_gif = None


def apply_default_style(style_url: str = DEFAULT_STYLE_URL) -> None:
    try:
        plt.style.use(style_url)
    except Exception:
        return


def ensure_output_dirs(img_dir: Path, scratch_dir: Path) -> None:
    img_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)


def maybe_optimize_gif(gif_path: Path, optimize: bool = True) -> None:
    if not optimize:
        return
    if _optimize_gif is None:
        return
    try:
        _optimize_gif(str(gif_path))
    except Exception:
        return


def _slow_duration_ms(duration_ms: int | float) -> int:
    return max(1, int(round(duration_ms * SLOWDOWN_FACTOR)))


def _slow_fps(fps: int | float) -> float:
    return max(0.1, float(fps) / SLOWDOWN_FACTOR)


@gif.frame
def _sine_bead_frame(step: int, total_steps: int, dpi: int = 140) -> None:
    plt.close("all")
    fig, axis = plt.subplots(figsize=(5, 3), dpi=dpi)
    x_values = np.linspace(0, 2 * np.pi, total_steps)
    y_values = np.sin(x_values)
    marker_index = step % total_steps

    axis.plot(x_values, y_values, color="#1f77b4", lw=2)
    axis.scatter([x_values[marker_index]], [y_values[marker_index]], color="#d62728", s=50)
    axis.set_title(f"Sine Bead — Scene {step + 1}")
    axis.set_xlabel("x")
    axis.set_ylabel("sin(x)")
    axis.set_ylim(-1.2, 1.2)
    axis.grid(alpha=0.3)
    plt.tight_layout()


def render_sine_bead(
    output_path: Path,
    frame_count: int = 24,
    hold_last: int = 6,
    duration_ms: int = 100,
    optimize: bool = True,
) -> Path:
    frames = [_sine_bead_frame(step, frame_count) for step in range(frame_count)]
    frames.extend(
        [_sine_bead_frame(frame_count - 1, frame_count) for _ in range(max(0, hold_last))]
    )
    gif.save(frames, str(output_path), duration=_slow_duration_ms(duration_ms))
    maybe_optimize_gif(output_path, optimize=optimize)
    return output_path


@gif.frame
def _nn_sigmoid_frame(step: int, total_steps: int, dpi: int = 140) -> None:
    plt.close("all")
    fig, axis = plt.subplots(figsize=(5, 3), dpi=dpi)
    x_values = np.linspace(-8, 8, 250)
    shift = np.interp(step, [0, max(1, total_steps - 1)], [-4, 4])
    y_values = 1 / (1 + np.exp(-(x_values - shift)))

    axis.plot(x_values, y_values, color="#2ca02c", lw=2)
    axis.set_title(f"NN Activation Shift — Step {step + 1}")
    axis.set_xlabel("Input")
    axis.set_ylabel("Sigmoid Output")
    axis.set_ylim(-0.05, 1.05)
    axis.grid(alpha=0.3)
    plt.tight_layout()


def render_nn_sigmoid_shift(
    output_path: Path,
    frame_count: int = 28,
    hold_last: int = 6,
    duration_ms: int = 90,
    optimize: bool = True,
) -> Path:
    frames = [_nn_sigmoid_frame(step, frame_count) for step in range(frame_count)]
    frames.extend(
        [_nn_sigmoid_frame(frame_count - 1, frame_count) for _ in range(max(0, hold_last))]
    )
    gif.save(frames, str(output_path), duration=_slow_duration_ms(duration_ms))
    maybe_optimize_gif(output_path, optimize=optimize)
    return output_path


@gif.frame
def _block_catalog_frame(
    title: str,
    subtitle: str,
    blocks: tuple[str, ...],
    active_index: int,
    footer: str,
    catalog_type: str,
    dpi: int = 170,
) -> None:
    plt.close("all")
    fig = plt.figure(figsize=(10.8, 5.2), dpi=dpi)
    grid = fig.add_gridspec(1, 2, width_ratios=[0.42, 0.58], wspace=0.06)
    list_axis = fig.add_subplot(grid[0, 0])
    concept_axis = fig.add_subplot(grid[0, 1])

    list_axis.set_xlim(0, 1)
    list_axis.set_ylim(0, 1)
    list_axis.axis("off")
    list_axis.set_facecolor("#f8fafc")

    list_axis.text(
        0.03, 0.965, title, fontsize=15, fontweight="bold", va="top", color="#0f172a"
    )
    list_axis.text(0.03, 0.905, subtitle, fontsize=10.5, va="top", color="#334155")

    start_y = 0.84
    row_gap = 0.062
    for index, block_name in enumerate(blocks):
        row_y = start_y - (index * row_gap)
        is_active = index == active_index
        list_axis.text(
            0.04,
            row_y,
            f"{index + 1:02d}. {block_name}",
            fontsize=9.2,
            va="center",
            color="#ffffff" if is_active else "#1e293b",
            bbox={
                "boxstyle": "round,pad=0.24",
                "facecolor": "#2563eb" if is_active else "#e2e8f0",
                "edgecolor": "none",
            },
        )

    block_name = blocks[active_index]
    if catalog_type == "dsp":
        _draw_dsp_block_concept(concept_axis, block_name, frame_seed=active_index)
    else:
        _draw_ml_block_concept(concept_axis, block_name, frame_seed=active_index)

    list_axis.text(0.03, 0.03, footer, fontsize=8.7, color="#64748b")
    fig.subplots_adjust(left=0.03, right=0.99, top=0.96, bottom=0.07, wspace=0.06)


def _style_concept_axis(axis: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    axis.set_title(title, fontsize=12, pad=8, color="#0f172a")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)


def _draw_dsp_block_concept(axis: plt.Axes, block_name: str, frame_seed: int) -> None:
    rng = np.random.default_rng(100 + frame_seed)

    if block_name == "Raw Data":
        x_values = np.linspace(0, 2.0, 300)
        y_values = (
            np.sin(2 * np.pi * 4 * x_values)
            + 0.45 * np.sin(2 * np.pi * 11 * x_values)
            + 0.15 * rng.normal(size=x_values.size)
        )
        axis.plot(x_values, y_values, color="#0ea5e9", lw=1.6)
        axis.scatter(x_values[::10], y_values[::10], color="#1d4ed8", s=8, alpha=0.7)
        _style_concept_axis(axis, "Raw Data: sensor waveform", "Time", "Amplitude")
        return

    if block_name == "Flatten":
        x_values = np.arange(0, 90)
        signal = 24 + 0.02 * x_values + 0.9 * np.sin(x_values / 11) + 0.25 * rng.normal(size=x_values.size)
        window = 9
        kernel = np.ones(window) / window
        smooth = np.convolve(signal, kernel, mode="same")
        rolling_var = np.convolve((signal - smooth) ** 2, kernel, mode="same")
        std = np.sqrt(np.maximum(rolling_var, 1e-6))
        axis.plot(x_values, signal, color="#64748b", lw=1.2, label="raw")
        axis.plot(x_values, smooth, color="#0f766e", lw=2.0, label="mean")
        axis.fill_between(x_values, smooth - std, smooth + std, color="#5eead4", alpha=0.28)
        axis.legend(loc="upper left", fontsize=8, frameon=False)
        _style_concept_axis(axis, "Flatten: statistical features", "Window", "Value")
        return

    if block_name == "Image":
        grid = np.linspace(-1, 1, 64)
        xx, yy = np.meshgrid(grid, grid)
        image = np.exp(-3 * (xx**2 + yy**2)) + 0.25 * np.sin(6 * xx) * np.cos(6 * yy)
        axis.imshow(image, cmap="viridis", origin="lower")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title("Image: pixel-space representation", fontsize=12, color="#0f172a")
        return

    if block_name == "Spectral features":
        sample_rate = 100
        t_values = np.arange(0, 1.0, 1 / sample_rate)
        waveform = np.sin(2 * np.pi * 7 * t_values) + 0.6 * np.sin(2 * np.pi * 18 * t_values)
        spectrum = np.abs(np.fft.rfft(waveform))
        freqs = np.fft.rfftfreq(t_values.size, d=1 / sample_rate)
        axis.bar(freqs, spectrum, width=0.9, color="#7c3aed", alpha=0.85)
        axis.set_xlim(0, 45)
        _style_concept_axis(axis, "Spectral features: FFT peaks", "Frequency (Hz)", "Magnitude")
        return

    if block_name == "Spectrogram":
        t_values = np.linspace(0, 1, 80)
        f_values = np.linspace(0, 1, 64)
        spec = (
            np.exp(-((f_values[:, None] - (0.2 + 0.5 * t_values[None, :])) ** 2) / 0.008)
            + 0.5 * np.exp(-((f_values[:, None] - (0.7 - 0.35 * t_values[None, :])) ** 2) / 0.015)
        )
        axis.imshow(spec, cmap="magma", origin="lower", aspect="auto")
        _style_concept_axis(axis, "Spectrogram: time-frequency map", "Time bins", "Frequency bins")
        return

    if block_name == "Audio MFE":
        mel_bands = 20
        frames = 32
        band_axis = np.linspace(0, 1, mel_bands)[:, None]
        frame_axis = np.linspace(0, 1, frames)[None, :]
        mfe = np.maximum(0, np.sin(3.2 * np.pi * band_axis + 2.5 * frame_axis) + 0.3)
        mfe += 0.25 * np.cos(2.0 * np.pi * frame_axis)
        axis.imshow(mfe, cmap="cividis", origin="lower", aspect="auto")
        _style_concept_axis(axis, "Audio MFE: Mel-scale energies", "Frames", "Mel bands")
        return

    if block_name == "Audio MFCC":
        coeff_axis = np.arange(13)
        coeff_values = np.cos(coeff_axis / 2.1 + frame_seed * 0.18) * np.exp(-coeff_axis / 9)
        axis.bar(coeff_axis, coeff_values, color="#0ea5e9")
        axis.axhline(0, color="#475569", lw=1)
        _style_concept_axis(axis, "Audio MFCC: cepstral coefficients", "Coefficient index", "Value")
        return

    if block_name == "Audio Syntiant":
        feature_map = rng.uniform(0, 1, size=(10, 24))
        axis.imshow(feature_map, cmap="plasma", origin="lower", aspect="auto")
        _style_concept_axis(axis, "Audio Syntiant: compact feature map", "Frame", "Feature")
        return

    if block_name == "IMU Syntiant":
        t_values = np.linspace(0, 2.2, 220)
        x_axis = np.sin(2.8 * t_values) + 0.1 * rng.normal(size=t_values.size)
        y_axis = 0.8 * np.cos(2.2 * t_values + 0.7) + 0.1 * rng.normal(size=t_values.size)
        z_axis = 0.6 * np.sin(4.0 * t_values + 1.4) + 0.1 * rng.normal(size=t_values.size)
        axis.plot(t_values, x_axis, label="acc_x", lw=1.4)
        axis.plot(t_values, y_axis, label="acc_y", lw=1.4)
        axis.plot(t_values, z_axis, label="acc_z", lw=1.4)
        axis.legend(loc="upper right", fontsize=8, frameon=False)
        _style_concept_axis(axis, "IMU Syntiant: multi-axis motion", "Time", "Acceleration")
        return

    if block_name == "HR/HRV features":
        beats = np.arange(0, 40)
        rr_intervals = 0.82 + 0.05 * np.sin(beats / 4.2) + 0.03 * rng.normal(size=beats.size)
        hr = 60 / np.clip(rr_intervals, 0.4, None)
        axis.plot(beats, hr, color="#dc2626", lw=1.8, label="Heart rate")
        axis.scatter(beats, hr, color="#b91c1c", s=10)
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
        axis.text(0.98, 0.92, f"RMSSD: {rmssd:.3f}", transform=axis.transAxes, ha="right", fontsize=9)
        axis.legend(loc="lower right", fontsize=8, frameon=False)
        _style_concept_axis(axis, "HR/HRV features", "Beat index", "BPM")
        return

    axis.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), color="#64748b")
    _style_concept_axis(axis, block_name)


def _draw_ml_block_concept(axis: plt.Axes, block_name: str, frame_seed: int) -> None:
    rng = np.random.default_rng(500 + frame_seed)

    if block_name == "Classification (Keras)":
        class_a = rng.normal(loc=(-1.0, -0.3), scale=0.35, size=(55, 2))
        class_b = rng.normal(loc=(0.9, 0.8), scale=0.35, size=(55, 2))
        axis.scatter(class_a[:, 0], class_a[:, 1], color="#2563eb", s=15, alpha=0.7, label="Class A")
        axis.scatter(class_b[:, 0], class_b[:, 1], color="#16a34a", s=15, alpha=0.7, label="Class B")
        x_values = np.linspace(-2.2, 2.2, 120)
        boundary = 0.55 * x_values + 0.1
        axis.plot(x_values, boundary, "--", color="#111827", lw=1.6, label="Decision boundary")
        axis.legend(loc="upper left", fontsize=8, frameon=False)
        _style_concept_axis(axis, "Classification: learned boundary", "Feature 1", "Feature 2")
        axis.set_xlim(-2.2, 2.2)
        axis.set_ylim(-1.9, 2.1)
        return

    if block_name == "Regression (Keras)":
        x_train = np.linspace(-2.5, 2.5, 45)
        y_train = 0.55 * (x_train**2) + 0.4 * x_train + 0.8 + 0.45 * rng.normal(size=x_train.size)
        coeff = np.polyfit(x_train, y_train, deg=2)
        x_pred = np.linspace(-2.7, 2.7, 220)
        y_pred = np.polyval(coeff, x_pred)
        axis.scatter(x_train, y_train, s=15, color="#64748b", alpha=0.7, label="Samples")
        axis.plot(x_pred, y_pred, color="#0ea5e9", lw=2.1, label="Model fit")
        axis.legend(loc="upper left", fontsize=8, frameon=False)
        _style_concept_axis(axis, "Regression: fit continuous value", "Input", "Target")
        return

    if block_name == "Anomaly Detection (K-means)":
        clusters = [
            (np.array([-1.2, -0.4]), "#2563eb"),
            (np.array([0.4, 1.0]), "#14b8a6"),
            (np.array([1.25, -0.8]), "#f59e0b"),
        ]
        for centroid, color in clusters:
            points = centroid + rng.normal(scale=0.22, size=(45, 2))
            axis.scatter(points[:, 0], points[:, 1], s=13, color=color, alpha=0.55)
            axis.scatter([centroid[0]], [centroid[1]], marker="x", s=80, color="#0f172a")
            axis.add_patch(Circle((centroid[0], centroid[1]), radius=0.58, fill=False, ls="--", lw=1.8, ec=color, alpha=0.9))
        anomaly = np.array([2.1, 1.8])
        axis.scatter([anomaly[0]], [anomaly[1]], marker="*", s=130, color="#ef4444")
        axis.annotate("Anomaly", xy=anomaly, xytext=(1.2, 2.1), arrowprops={"arrowstyle": "->", "color": "#ef4444"}, color="#ef4444", fontsize=9)
        _style_concept_axis(axis, "K-means: spherical clusters + distance threshold", "Feature 1", "Feature 2")
        axis.set_xlim(-2.2, 2.5)
        axis.set_ylim(-1.8, 2.4)
        axis.set_aspect("equal", adjustable="box")
        return

    if block_name == "Anomaly Detection (GMM)":
        components = [
            ((-1.1, -0.35), 0.62, 0.26, 20, "#2563eb"),
            ((0.45, 0.95), 0.88, 0.30, -35, "#14b8a6"),
            ((1.3, -0.75), 0.70, 0.22, 50, "#f59e0b"),
        ]
        for (mean_x, mean_y), major, minor, angle, color in components:
            theta = np.deg2rad(angle)
            rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            covariance = rotation @ np.diag([major**2, minor**2]) @ rotation.T
            points = rng.multivariate_normal([mean_x, mean_y], covariance, size=48)
            axis.scatter(points[:, 0], points[:, 1], s=13, color=color, alpha=0.50)
            for scale, alpha in ((1.0, 0.9), (1.7, 0.55), (2.4, 0.30)):
                axis.add_patch(
                    Ellipse(
                        xy=(mean_x, mean_y),
                        width=2 * major * scale,
                        height=2 * minor * scale,
                        angle=angle,
                        fill=False,
                        lw=1.5,
                        ec=color,
                        alpha=alpha,
                    )
                )
        anomaly = np.array([2.2, 1.9])
        axis.scatter([anomaly[0]], [anomaly[1]], marker="*", s=130, color="#ef4444")
        axis.annotate("Low probability", xy=anomaly, xytext=(1.25, 2.15), arrowprops={"arrowstyle": "->", "color": "#ef4444"}, color="#ef4444", fontsize=9)
        _style_concept_axis(axis, "GMM: elliptical clusters + probability score", "Feature 1", "Feature 2")
        axis.set_xlim(-2.2, 2.6)
        axis.set_ylim(-1.9, 2.5)
        axis.set_aspect("equal", adjustable="box")
        return

    if block_name == "Visual anomaly detection (FOMO-AD)":
        heat = rng.uniform(0.02, 0.2, size=(14, 14))
        row, col = 8, 10
        heat[row, col] = 1.0
        axis.imshow(heat, cmap="magma", origin="lower")
        axis.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, ec="#22c55e", lw=2.0))
        axis.set_title("FOMO-AD: pixel-level anomaly heatmap", fontsize=12, color="#0f172a")
        axis.set_xticks([])
        axis.set_yticks([])
        return

    if block_name == "Image Classification (Transfer Learning)":
        labels = ["gear", "bolt", "bearing", "other"]
        probs = np.array([0.08, 0.12, 0.73, 0.07])
        axis.barh(labels, probs, color=["#94a3b8", "#94a3b8", "#22c55e", "#94a3b8"])
        axis.set_xlim(0, 1)
        _style_concept_axis(axis, "Image classification probabilities", "Probability", "")
        return

    if block_name == "Keyword Spotting (Transfer Learning)":
        time = np.linspace(0, 1, 350)
        waveform = 0.5 * np.sin(2 * np.pi * 6 * time) + 0.25 * np.sin(2 * np.pi * 18 * time)
        waveform += 0.05 * rng.normal(size=time.size)
        axis.plot(time, waveform, color="#0ea5e9", lw=1.3)
        axis.axvspan(0.55, 0.75, color="#22c55e", alpha=0.25, label="keyword window")
        axis.legend(loc="upper left", fontsize=8, frameon=False)
        _style_concept_axis(axis, "Keyword spotting over audio stream", "Time", "Amplitude")
        axis.set_ylim(-1.2, 1.2)
        return

    if block_name == "Object Detection (MobileNetV2 SSD FPN)":
        canvas = np.tile(np.linspace(0.25, 0.75, 180), (110, 1))
        axis.imshow(canvas, cmap="gray", origin="lower")
        for x_pos, y_pos, width, height, label in [
            (18, 16, 45, 35, "part A"),
            (96, 42, 55, 45, "part B"),
        ]:
            axis.add_patch(Rectangle((x_pos, y_pos), width, height, fill=False, ec="#22c55e", lw=2))
            axis.text(x_pos, y_pos + height + 3, label, color="#22c55e", fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title("Object detection: bounding boxes", fontsize=12, color="#0f172a")
        return

    if block_name == "Object Detection (FOMO)":
        fmap = np.zeros((12, 12))
        fmap[4, 8] = 0.95
        fmap[8, 3] = 0.75
        axis.imshow(fmap, cmap="viridis", origin="lower", vmin=0, vmax=1)
        axis.set_xticks(np.arange(-0.5, 12, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, 12, 1), minor=True)
        axis.grid(which="minor", color="white", linestyle="-", linewidth=0.6, alpha=0.6)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title("FOMO: grid-cell object presence", fontsize=12, color="#0f172a")
        return

    if block_name == "Classical ML":
        xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 160), np.linspace(-2.0, 2.0, 150))
        regions = (yy > 0.45 * np.sin(1.3 * xx) - 0.2).astype(int) + (xx > 0.6).astype(int)
        axis.contourf(xx, yy, regions, levels=[-0.5, 0.5, 1.5, 2.5], colors=["#dbeafe", "#dcfce7", "#fef3c7"], alpha=0.85)
        points = rng.normal(size=(80, 2))
        axis.scatter(points[:, 0], points[:, 1], s=10, color="#0f172a", alpha=0.35)
        _style_concept_axis(axis, "Classical ML: partitioned feature space", "Feature 1", "Feature 2")
        return

    if block_name == "Custom block (PyTorch/Keras/scikit-learn)":
        axis.axis("off")
        axis.set_title("Custom learning block pipeline", fontsize=12, pad=8, color="#0f172a")
        boxes = [
            (0.06, 0.42, 0.23, 0.18, "Features"),
            (0.38, 0.42, 0.27, 0.18, "Custom Model\n(PyTorch/Keras)"),
            (0.74, 0.42, 0.20, 0.18, "Output"),
        ]
        for x_pos, y_pos, width, height, label in boxes:
            axis.add_patch(Rectangle((x_pos, y_pos), width, height, fc="#e2e8f0", ec="#64748b", lw=1.3, transform=axis.transAxes))
            axis.text(x_pos + width / 2, y_pos + height / 2, label, ha="center", va="center", fontsize=9, transform=axis.transAxes)
        axis.annotate("", xy=(0.38, 0.51), xytext=(0.29, 0.51), arrowprops={"arrowstyle": "->", "lw": 1.6}, xycoords=axis.transAxes)
        axis.annotate("", xy=(0.74, 0.51), xytext=(0.65, 0.51), arrowprops={"arrowstyle": "->", "lw": 1.6}, xycoords=axis.transAxes)
        return

    axis.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), color="#64748b")
    _style_concept_axis(axis, block_name)


def render_block_catalog_animation(
    output_path: Path,
    title: str,
    subtitle: str,
    blocks: Sequence[str],
    footer: str,
    catalog_type: str,
    repeat_each: int = 2,
    hold_last: int = 8,
    duration_ms: int = 130,
    optimize: bool = True,
) -> Path:
    block_tuple = tuple(blocks)
    if not block_tuple:
        raise ValueError("blocks must contain at least one item")

    frames = []
    for index in range(len(block_tuple)):
        frames.extend(
            [
                _block_catalog_frame(
                    title,
                    subtitle,
                    block_tuple,
                    index,
                    footer,
                    catalog_type,
                )
                for _ in range(max(1, repeat_each))
            ]
        )
    frames.extend(
        [
            _block_catalog_frame(
                title,
                subtitle,
                block_tuple,
                len(block_tuple) - 1,
                footer,
                catalog_type,
            )
            for _ in range(max(0, hold_last))
        ]
    )

    gif.save(frames, str(output_path), duration=_slow_duration_ms(duration_ms))
    maybe_optimize_gif(output_path, optimize=optimize)
    return output_path


def render_dsp_processing_blocks(
    output_path: Path,
    repeat_each: int = 2,
    hold_last: int = 8,
    duration_ms: int = 130,
    optimize: bool = True,
) -> Path:
    return render_block_catalog_animation(
        output_path=output_path,
        title="Edge Impulse Processing Blocks",
        subtitle="Extract meaningful features from sensor data",
        blocks=DSP_PROCESSING_BLOCKS,
        footer="Source: Edge Impulse processing blocks GitHub repository",
        catalog_type="dsp",
        repeat_each=repeat_each,
        hold_last=hold_last,
        duration_ms=duration_ms,
        optimize=optimize,
    )


def render_ml_learning_blocks(
    output_path: Path,
    repeat_each: int = 2,
    hold_last: int = 8,
    duration_ms: int = 130,
    optimize: bool = True,
) -> Path:
    return render_block_catalog_animation(
        output_path=output_path,
        title="Edge Impulse Learning Blocks",
        subtitle="Train models after feature extraction",
        blocks=ML_LEARNING_BLOCKS,
        footer="You can also build custom blocks with PyTorch, Keras, or scikit-learn",
        catalog_type="ml",
        repeat_each=repeat_each,
        hold_last=hold_last,
        duration_ms=duration_ms,
        optimize=optimize,
    )


def _slugify_block_name(block_name: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in block_name)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


@gif.frame
def _single_block_concept_frame(
    block_name: str,
    catalog_type: str,
    step: int,
    total_steps: int,
    dpi: int = 170,
) -> None:
    plt.close("all")
    fig, concept_axis = plt.subplots(figsize=(8.0, 4.8), dpi=dpi)

    if catalog_type == "dsp":
        title = f"DSP Processing Block: {block_name}"
        subtitle = "Feature extraction concept illustration"
        footer = "Edge Impulse processing blocks"
        _draw_dsp_block_concept(concept_axis, block_name, frame_seed=step)
    else:
        title = f"ML Learning Block: {block_name}"
        subtitle = "Model-training concept illustration"
        footer = "Edge Impulse learning blocks"
        _draw_ml_block_concept(concept_axis, block_name, frame_seed=step)

    fig.suptitle(title, x=0.02, y=0.99, ha="left", fontsize=14, color="#0f172a")
    fig.text(0.02, 0.94, subtitle, ha="left", fontsize=9.8, color="#334155")
    fig.text(0.02, 0.02, footer, ha="left", fontsize=8.8, color="#64748b")
    fig.text(0.98, 0.02, f"Frame {step + 1}/{total_steps}", ha="right", fontsize=8.8, color="#64748b")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.13)


def render_single_block_animation(
    output_path: Path,
    block_name: str,
    catalog_type: str,
    frame_count: int = 18,
    hold_last: int = 6,
    duration_ms: int = 120,
    optimize: bool = True,
) -> Path:
    if catalog_type not in {"dsp", "ml"}:
        raise ValueError("catalog_type must be 'dsp' or 'ml'")

    frames = [
        _single_block_concept_frame(
            block_name=block_name,
            catalog_type=catalog_type,
            step=step,
            total_steps=frame_count,
        )
        for step in range(frame_count)
    ]
    frames.extend(
        [
            _single_block_concept_frame(
                block_name=block_name,
                catalog_type=catalog_type,
                step=frame_count - 1,
                total_steps=frame_count,
            )
            for _ in range(max(0, hold_last))
        ]
    )

    gif.save(frames, str(output_path), duration=_slow_duration_ms(duration_ms))
    maybe_optimize_gif(output_path, optimize=optimize)
    return output_path


def render_all_dsp_processing_block_animations(
    img_dir: Path,
    frame_count: int = 18,
    hold_last: int = 6,
    duration_ms: int = 120,
    optimize: bool = True,
) -> list[Path]:
    outputs: list[Path] = []
    for block_name in DSP_PROCESSING_BLOCKS:
        output_path = img_dir / f"dsp_block_{_slugify_block_name(block_name)}.gif"
        outputs.append(
            render_single_block_animation(
                output_path=output_path,
                block_name=block_name,
                catalog_type="dsp",
                frame_count=frame_count,
                hold_last=hold_last,
                duration_ms=duration_ms,
                optimize=optimize,
            )
        )
    return outputs


def render_all_ml_learning_block_animations(
    img_dir: Path,
    frame_count: int = 18,
    hold_last: int = 6,
    duration_ms: int = 120,
    optimize: bool = True,
) -> list[Path]:
    outputs: list[Path] = []
    for block_name in ML_LEARNING_BLOCKS:
        output_path = img_dir / f"ml_block_{_slugify_block_name(block_name)}.gif"
        outputs.append(
            render_single_block_animation(
                output_path=output_path,
                block_name=block_name,
                catalog_type="ml",
                frame_count=frame_count,
                hold_last=hold_last,
                duration_ms=duration_ms,
                optimize=optimize,
            )
        )
    return outputs


def _save_dsp_sine_frame(frame_path: Path, step: int, total_steps: int, dpi: int = 140) -> None:
    plt.close("all")
    fig, axis = plt.subplots(figsize=(5, 3), dpi=dpi)

    x_values = np.linspace(0, 2 * np.pi, 200)
    phase = np.interp(step, [0, max(1, total_steps - 1)], [0, 2 * np.pi])
    y_values = np.sin(x_values + phase)

    axis.plot(x_values, y_values, color="#9467bd", lw=2)
    axis.set_title(f"DSP Phase Shift — Frame {step + 1}")
    axis.set_xlabel("Time")
    axis.set_ylabel("Amplitude")
    axis.set_ylim(-1.2, 1.2)
    axis.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(frame_path)
    plt.close(fig)


def render_dsp_sine_shift(
    output_path: Path,
    scratch_dir: Path,
    frame_count: int = 24,
    fps: int = 8,
    optimize: bool = True,
) -> Path:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    for existing in scratch_dir.glob("dsp_sine_shift_*.png"):
        existing.unlink(missing_ok=True)

    frame_paths: list[Path] = []
    for step in range(frame_count):
        frame_path = scratch_dir / f"dsp_sine_shift_{step:03d}.png"
        _save_dsp_sine_frame(frame_path, step=step, total_steps=frame_count)
        frame_paths.append(frame_path)

    frame_duration_ms = 1000 / _slow_fps(max(1, fps))
    with imageio.get_writer(str(output_path), mode="I", duration=frame_duration_ms) as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))

    maybe_optimize_gif(output_path, optimize=optimize)
    return output_path


def load_gapminder_data(csv_url: str = DEFAULT_GAPMINDER_URL) -> pd.DataFrame:
    data_frame = pd.read_csv(csv_url)
    required_columns = {
        "Continent",
        "Year",
        "GDP per capita",
        "Life expectancy",
        "Population",
    }
    missing = required_columns.difference(data_frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return data_frame


def build_colour_mapping(data_frame: pd.DataFrame) -> dict[str, str]:
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
    continents = sorted(data_frame["Continent"].dropna().unique())
    return {
        continent: palette[index % len(palette)]
        for index, continent in enumerate(continents)
    }


@gif.frame
def _gapminder_frame(
    year: int,
    year_frame: pd.DataFrame,
    x_max: float,
    y_max: float,
    dpi: int = 160,
) -> None:
    plt.close("all")
    fig, axis = plt.subplots(figsize=(6, 4), dpi=dpi)
    axis.scatter(
        x=year_frame["GDP per capita"] / 1e3,
        y=year_frame["Life expectancy"],
        c=year_frame["colour"],
        s=year_frame["Population"] / 1e6,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.2,
    )
    axis.set_xlim(0, x_max)
    axis.set_ylim(20, y_max)
    axis.set_xlabel("GDP per capita (thousands, 2011 USD)")
    axis.set_ylabel("Life expectancy (years)")
    axis.set_title(f"Gapminder — {year}")
    axis.grid(alpha=0.2)
    plt.tight_layout()


def render_gapminder_full(
    output_path: Path,
    data_frame: pd.DataFrame | None = None,
    start_year: int | None = None,
    end_year: int = 2018,
    hold_last: int = 8,
    duration_ms: int = 120,
    optimize: bool = True,
) -> Path:
    frame = data_frame.copy() if data_frame is not None else load_gapminder_data()
    frame = frame[frame["Year"] <= end_year].copy()
    if start_year is not None:
        frame = frame[frame["Year"] >= start_year].copy()
    if frame.empty:
        raise ValueError("No Gapminder rows available for selected year range.")

    colour_mapping = build_colour_mapping(frame)
    frame["colour"] = frame["Continent"].map(colour_mapping)
    years = sorted(int(year) for year in frame["Year"].unique())
    x_max = max(100.0, float((frame["GDP per capita"] / 1e3).max()) * 1.05)
    y_max = float(frame["Life expectancy"].max()) * 1.1

    frames = [
        _gapminder_frame(year, frame[frame["Year"] == year], x_max=x_max, y_max=y_max)
        for year in years
    ]
    frames.extend(
        [
            _gapminder_frame(
                years[-1], frame[frame["Year"] == years[-1]], x_max=x_max, y_max=y_max
            )
            for _ in range(max(0, hold_last))
        ]
    )

    gif.save(frames, str(output_path), duration=_slow_duration_ms(duration_ms))
    maybe_optimize_gif(output_path, optimize=optimize)
    return output_path


def run_selected_presets(
    presets: Iterable[str],
    img_dir: Path,
    scratch_dir: Path,
    optimize: bool,
    fps: int,
) -> list[Path]:
    ensure_output_dirs(img_dir, scratch_dir)
    apply_default_style()

    generated: list[Path] = []
    for preset in presets:
        output_path = img_dir / f"{preset}.gif"
        if preset == "dsp_sine_shift":
            generated.append(
                render_dsp_sine_shift(
                    output_path=output_path,
                    scratch_dir=scratch_dir,
                    frame_count=24,
                    fps=fps,
                    optimize=optimize,
                )
            )
        elif preset == "dsp_processing_blocks":
            generated.append(
                render_dsp_processing_blocks(
                    output_path=output_path,
                    repeat_each=2,
                    hold_last=8,
                    duration_ms=130,
                    optimize=optimize,
                )
            )
        elif preset == "dsp_processing_blocks_individual":
            generated.extend(
                render_all_dsp_processing_block_animations(
                    img_dir=img_dir,
                    frame_count=18,
                    hold_last=6,
                    duration_ms=120,
                    optimize=optimize,
                )
            )
        elif preset == "nn_sigmoid_shift":
            generated.append(
                render_nn_sigmoid_shift(
                    output_path=output_path,
                    frame_count=28,
                    hold_last=6,
                    duration_ms=90,
                    optimize=optimize,
                )
            )
        elif preset == "ml_learning_blocks":
            generated.append(
                render_ml_learning_blocks(
                    output_path=output_path,
                    repeat_each=2,
                    hold_last=8,
                    duration_ms=130,
                    optimize=optimize,
                )
            )
        elif preset == "ml_learning_blocks_individual":
            generated.extend(
                render_all_ml_learning_block_animations(
                    img_dir=img_dir,
                    frame_count=18,
                    hold_last=6,
                    duration_ms=120,
                    optimize=optimize,
                )
            )
        elif preset == "sine_bead":
            generated.append(
                render_sine_bead(
                    output_path=output_path,
                    frame_count=24,
                    hold_last=6,
                    duration_ms=100,
                    optimize=optimize,
                )
            )
        elif preset == "gapminder_full":
            generated.append(
                render_gapminder_full(
                    output_path=output_path,
                    end_year=2018,
                    hold_last=8,
                    duration_ms=120,
                    optimize=optimize,
                )
            )
        else:
            raise ValueError(f"Unknown preset: {preset}")
    return generated


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ML/DSP concept animations as GIFs."
    )
    parser.add_argument("--preset", choices=PRESET_NAMES, help="Generate one preset")
    parser.add_argument("--all", action="store_true", help="Generate all presets")
    parser.add_argument("--img-dir", default="img", help="Output GIF directory")
    parser.add_argument(
        "--scratch-dir",
        default="scratch",
        help="Temporary frame directory for image-based presets",
    )
    parser.add_argument("--fps", type=int, default=8, help="FPS for imageio presets")
    parser.add_argument(
        "--skip-optimize",
        action="store_true",
        help="Skip gifsicle optimization step",
    )
    args = parser.parse_args(argv)

    if not args.all and args.preset is None:
        parser.error("Choose either --all or --preset <name>.")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    presets = DEFAULT_ALL_PRESETS if args.all else (args.preset,)
    output_paths = run_selected_presets(
        presets=presets,
        img_dir=Path(args.img_dir),
        scratch_dir=Path(args.scratch_dir),
        optimize=not args.skip_optimize,
        fps=args.fps,
    )
    for output_path in output_paths:
        print(f"Generated: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())