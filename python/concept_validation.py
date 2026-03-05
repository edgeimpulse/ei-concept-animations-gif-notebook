from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
import matplotlib
import numpy as np
from matplotlib.patches import Circle, Ellipse, Rectangle

from animation_pipeline import (
    DSP_PROCESSING_BLOCKS,
    ML_LEARNING_BLOCKS,
    classification_decision_boundary,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REFERENCE_DPI = 170
REFERENCE_SIZE = 192


@dataclass(frozen=True)
class ConceptSpec:
    catalog_type: str
    block_name: str
    gif_name: str
    script_type: str


@dataclass(frozen=True)
class ConceptValidationResult:
    catalog_type: str
    block_name: str
    gif_path: str
    script_type: str
    accuracy_percent: float | None


@dataclass(frozen=True)
class GifStyleValidationResult:
    gif_name: str
    gif_path: str
    header_centered: bool
    watermark_bottom_left: bool
    palette_aligned: bool
    red_poi_present: bool
    requires_red_poi: bool
    style_score_percent: float


STYLE_GUIDE_SPEC: dict[str, object] = {
    "header": {
        "placement": "top_center",
        "description": "Centered header title in top band",
    },
    "watermark": {
        "placement": "bottom_left",
        "description": "Brand watermark in lower-left corner",
    },
    "palette_hex": [
        "#0f172a",  # base text
        "#334155",  # secondary text
        "#64748b",  # muted accents
        "#2563eb",  # primary blue
        "#10b981",  # flow highlight green
        "#8b5cf6",  # model/activation purple
        "#f59e0b",  # warm highlight
        "#dc2626",  # point-of-interest red
    ],
    "point_of_interest": {
        "color": "#dc2626",
        "usage": "Use red to highlight anomalies, detections, errors, or key focus regions",
    },
}

STYLE_CHECK_THRESHOLDS: dict[str, float] = {
    "header_activity_min": 0.010,
    "header_center_margin": 0.002,
    "watermark_activity_min": 0.008,
    "palette_alignment_min": 0.45,
    "red_poi_min": 0.0008,
}

RED_POI_FILENAME_HINTS: tuple[str, ...] = (
    "anomaly",
    "detection",
    "fomo",
    "transfer_learning",
    "quantization",
    "backpropagation",
)


def _hex_to_rgb(hex_color: str) -> np.ndarray:
    cleaned = hex_color.lstrip("#")
    if len(cleaned) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return np.array([int(cleaned[0:2], 16), int(cleaned[2:4], 16), int(cleaned[4:6], 16)], dtype=np.float32)


def _ensure_rgb(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        return np.repeat(array[..., None], 3, axis=2)
    if array.ndim == 3 and array.shape[2] >= 3:
        return array[..., :3]
    raise ValueError("Unsupported frame shape for RGB conversion")


def _extract_region(frame_rgb: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    height, width, _ = frame_rgb.shape
    left = max(0, min(width - 1, int(np.floor(x0 * width))))
    right = max(left + 1, min(width, int(np.ceil(x1 * width))))
    top = max(0, min(height - 1, int(np.floor(y0 * height))))
    bottom = max(top + 1, min(height, int(np.ceil(y1 * height))))
    return frame_rgb[top:bottom, left:right]


def _estimate_background_rgb(frame_rgb: np.ndarray) -> np.ndarray:
    height, width, _ = frame_rgb.shape
    patch_h = max(2, int(round(height * 0.08)))
    patch_w = max(2, int(round(width * 0.08)))

    corners = [
        frame_rgb[:patch_h, :patch_w],
        frame_rgb[:patch_h, -patch_w:],
        frame_rgb[-patch_h:, :patch_w],
        frame_rgb[-patch_h:, -patch_w:],
    ]
    stacked = np.concatenate([corner.reshape(-1, 3) for corner in corners], axis=0)
    return np.median(stacked.astype(np.float32), axis=0)


def _region_activity_ratio(region_rgb: np.ndarray, background_rgb: np.ndarray, tolerance: float = 20.0) -> float:
    distances = np.linalg.norm(region_rgb.astype(np.float32) - background_rgb[None, None, :], axis=2)
    return float(np.mean(distances > tolerance))


def _header_centered(frame_rgb: np.ndarray, background_rgb: np.ndarray) -> bool:
    top_center = _extract_region(frame_rgb, 0.30, 0.70, 0.00, 0.18)
    top_left = _extract_region(frame_rgb, 0.02, 0.26, 0.00, 0.18)
    top_right = _extract_region(frame_rgb, 0.74, 0.98, 0.00, 0.18)

    center_activity = _region_activity_ratio(top_center, background_rgb)
    side_activity = 0.5 * (
        _region_activity_ratio(top_left, background_rgb) + _region_activity_ratio(top_right, background_rgb)
    )

    return (
        center_activity >= STYLE_CHECK_THRESHOLDS["header_activity_min"]
        and (center_activity - side_activity) >= STYLE_CHECK_THRESHOLDS["header_center_margin"]
    )


def _watermark_bottom_left(frame_rgb: np.ndarray, background_rgb: np.ndarray) -> bool:
    bottom_left = _extract_region(frame_rgb, 0.02, 0.34, 0.86, 1.00)
    activity = _region_activity_ratio(bottom_left, background_rgb)
    return activity >= STYLE_CHECK_THRESHOLDS["watermark_activity_min"]


def _palette_alignment(frame_rgb: np.ndarray, palette_rgb: np.ndarray) -> float:
    sampled = frame_rgb[::3, ::3].reshape(-1, 3).astype(np.float32)
    distances = np.linalg.norm(sampled[:, None, :] - palette_rgb[None, :, :], axis=2)
    min_dist = np.min(distances, axis=1)
    return float(np.mean(min_dist <= 95.0))


def _red_poi_ratio(frame_rgb: np.ndarray) -> float:
    red = frame_rgb[..., 0].astype(np.float32)
    green = frame_rgb[..., 1].astype(np.float32)
    blue = frame_rgb[..., 2].astype(np.float32)
    mask = (red > 145.0) & ((red - green) > 45.0) & ((red - blue) > 45.0)
    return float(np.mean(mask))


def _requires_red_poi(gif_name: str) -> bool:
    lowered = gif_name.lower()
    return any(hint in lowered for hint in RED_POI_FILENAME_HINTS)


def _sample_gif_frames(gif_path: Path, max_frames: int = 5) -> list[np.ndarray]:
    frames = imageio.mimread(gif_path, memtest=False)
    if not frames:
        raise ValueError(f"No frames found in GIF: {gif_path}")
    sample_count = min(max_frames, len(frames))
    indices = np.linspace(0, len(frames) - 1, sample_count).astype(np.int32)
    unique_indices = sorted(set(int(index) for index in indices))
    return [np.asarray(frames[index]) for index in unique_indices]


def _slugify_block_name(block_name: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in block_name)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _script_type_for_concept(catalog_type: str, block_name: str) -> str:
    if catalog_type == "dsp":
        return "dsp"
    if "Keras" in block_name:
        return "keras"
    return "nn"


def build_concept_specs() -> list[ConceptSpec]:
    specs: list[ConceptSpec] = []
    for block_name in DSP_PROCESSING_BLOCKS:
        specs.append(
            ConceptSpec(
                catalog_type="dsp",
                block_name=block_name,
                gif_name=f"dsp_block_{_slugify_block_name(block_name)}.gif",
                script_type=_script_type_for_concept("dsp", block_name),
            )
        )
    for block_name in ML_LEARNING_BLOCKS:
        specs.append(
            ConceptSpec(
                catalog_type="ml",
                block_name=block_name,
                gif_name=f"ml_block_{_slugify_block_name(block_name)}.gif",
                script_type=_script_type_for_concept("ml", block_name),
            )
        )
    return specs


def _style_axis(axis: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    axis.set_title(title, fontsize=12, pad=8, color="#0f172a")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)


def _draw_reference_dsp(axis: plt.Axes, block_name: str, seed: int) -> None:
    rng = np.random.default_rng(900 + seed)

    if block_name == "Raw Data":
        x_values = np.linspace(0, 2.0, 300)
        y_values = (
            np.sin(2 * np.pi * 4 * x_values)
            + 0.45 * np.sin(2 * np.pi * 11 * x_values)
            + 0.1 * rng.normal(size=x_values.size)
        )
        axis.plot(x_values, y_values, color="#0ea5e9", lw=1.6)
        axis.scatter(x_values[::10], y_values[::10], color="#1d4ed8", s=8, alpha=0.7)
        _style_axis(axis, "Raw Data: sensor waveform", "Time", "Amplitude")
        return

    if block_name == "Flatten":
        x_values = np.arange(0, 90)
        signal = 24 + 0.02 * x_values + 0.8 * np.sin(x_values / 10.5) + 0.2 * rng.normal(size=x_values.size)
        kernel = np.ones(9) / 9
        smooth = np.convolve(signal, kernel, mode="same")
        rolling_var = np.convolve((signal - smooth) ** 2, kernel, mode="same")
        std = np.sqrt(np.maximum(rolling_var, 1e-6))
        axis.plot(x_values, signal, color="#64748b", lw=1.2, label="raw")
        axis.plot(x_values, smooth, color="#0f766e", lw=2.0, label="mean")
        axis.fill_between(x_values, smooth - std, smooth + std, color="#5eead4", alpha=0.28)
        axis.legend(loc="upper left", fontsize=8, frameon=False)
        _style_axis(axis, "Flatten: statistical features", "Window", "Value")
        return

    if block_name == "Image":
        grid = np.linspace(-1, 1, 64)
        xx, yy = np.meshgrid(grid, grid)
        image = np.exp(-3 * (xx**2 + yy**2)) + 0.2 * np.sin(5.5 * xx) * np.cos(5.5 * yy)
        axis.imshow(image, cmap="viridis", origin="lower")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title("Image: pixel-space representation", fontsize=12, color="#0f172a")
        return

    if block_name == "Spectral features":
        sample_rate = 100
        t_values = np.arange(0, 1.0, 1 / sample_rate)
        waveform = np.sin(2 * np.pi * 7 * t_values) + 0.55 * np.sin(2 * np.pi * 18 * t_values)
        spectrum = np.abs(np.fft.rfft(waveform))
        freqs = np.fft.rfftfreq(t_values.size, d=1 / sample_rate)
        axis.bar(freqs, spectrum, width=0.9, color="#7c3aed", alpha=0.85)
        axis.set_xlim(0, 45)
        _style_axis(axis, "Spectral features: FFT peaks", "Frequency (Hz)", "Magnitude")
        return

    if block_name == "Spectrogram":
        t_values = np.linspace(0, 1, 80)
        f_values = np.linspace(0, 1, 64)
        spec = (
            np.exp(-((f_values[:, None] - (0.2 + 0.5 * t_values[None, :])) ** 2) / 0.009)
            + 0.45 * np.exp(-((f_values[:, None] - (0.7 - 0.35 * t_values[None, :])) ** 2) / 0.016)
        )
        axis.imshow(spec, cmap="magma", origin="lower", aspect="auto")
        _style_axis(axis, "Spectrogram: time-frequency map", "Time bins", "Frequency bins")
        return

    if block_name == "Audio MFE":
        mel_bands = 20
        frames = 32
        band_axis = np.linspace(0, 1, mel_bands)[:, None]
        frame_axis = np.linspace(0, 1, frames)[None, :]
        mfe = np.maximum(0, np.sin(3.0 * np.pi * band_axis + 2.2 * frame_axis) + 0.3)
        mfe += 0.22 * np.cos(2.0 * np.pi * frame_axis)
        axis.imshow(mfe, cmap="cividis", origin="lower", aspect="auto")
        _style_axis(axis, "Audio MFE: Mel-scale energies", "Frames", "Mel bands")
        return

    if block_name == "Audio MFCC":
        coeff_axis = np.arange(13)
        coeff_values = np.cos(coeff_axis / 2.1 + seed * 0.18) * np.exp(-coeff_axis / 9)
        axis.bar(coeff_axis, coeff_values, color="#0ea5e9")
        axis.axhline(0, color="#475569", lw=1)
        _style_axis(axis, "Audio MFCC: cepstral coefficients", "Coefficient index", "Value")
        return

    if block_name == "Audio Syntiant":
        feature_map = rng.uniform(0, 1, size=(10, 24))
        axis.imshow(feature_map, cmap="plasma", origin="lower", aspect="auto")
        _style_axis(axis, "Audio Syntiant: compact feature map", "Frame", "Feature")
        return

    if block_name == "IMU Syntiant":
        t_values = np.linspace(0, 2.2, 220)
        x_axis = np.sin(2.8 * t_values) + 0.08 * rng.normal(size=t_values.size)
        y_axis = 0.8 * np.cos(2.2 * t_values + 0.7) + 0.08 * rng.normal(size=t_values.size)
        z_axis = 0.6 * np.sin(4.0 * t_values + 1.4) + 0.08 * rng.normal(size=t_values.size)
        axis.plot(t_values, x_axis, label="acc_x", lw=1.4)
        axis.plot(t_values, y_axis, label="acc_y", lw=1.4)
        axis.plot(t_values, z_axis, label="acc_z", lw=1.4)
        axis.legend(loc="upper right", fontsize=8, frameon=False)
        _style_axis(axis, "IMU Syntiant: multi-axis motion", "Time", "Acceleration")
        return

    if block_name == "HR/HRV features":
        beats = np.arange(0, 40)
        rr_intervals = 0.82 + 0.05 * np.sin(beats / 4.2) + 0.02 * rng.normal(size=beats.size)
        hr = 60 / np.clip(rr_intervals, 0.4, None)
        axis.plot(beats, hr, color="#dc2626", lw=1.8, label="Heart rate")
        axis.scatter(beats, hr, color="#b91c1c", s=10)
        axis.legend(loc="lower right", fontsize=8, frameon=False)
        _style_axis(axis, "HR/HRV features", "Beat index", "BPM")
        return

    axis.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), color="#64748b")
    _style_axis(axis, block_name)


def _draw_reference_ml(axis: plt.Axes, block_name: str, seed: int) -> None:
    rng = np.random.default_rng(1500 + seed)

    if block_name == "Classification (Keras)":
        class_a = rng.normal(loc=(-1.0, -0.3), scale=0.35, size=(55, 2))
        class_b = rng.normal(loc=(0.9, 0.8), scale=0.35, size=(55, 2))
        axis.scatter(class_a[:, 0], class_a[:, 1], color="#2563eb", s=15, alpha=0.7, label="Class A")
        axis.scatter(class_b[:, 0], class_b[:, 1], color="#16a34a", s=15, alpha=0.7, label="Class B")
        x_values = np.linspace(-2.2, 2.2, 120)
        axis.plot(
            x_values,
            classification_decision_boundary(x_values),
            "--",
            color="#111827",
            lw=1.6,
            label="Decision boundary",
        )
        axis.legend(loc="upper left", fontsize=8, frameon=False)
        _style_axis(axis, "Classification: learned boundary", "Feature 1", "Feature 2")
        axis.set_xlim(-2.2, 2.2)
        axis.set_ylim(-1.9, 2.1)
        return

    if block_name == "Regression (Keras)":
        x_train = np.linspace(-2.5, 2.5, 45)
        y_train = 0.55 * (x_train**2) + 0.4 * x_train + 0.8 + 0.4 * rng.normal(size=x_train.size)
        coeff = np.polyfit(x_train, y_train, deg=2)
        x_pred = np.linspace(-2.7, 2.7, 220)
        y_pred = np.polyval(coeff, x_pred)
        axis.scatter(x_train, y_train, s=15, color="#64748b", alpha=0.7, label="Samples")
        axis.plot(x_pred, y_pred, color="#0ea5e9", lw=2.1, label="Model fit")
        axis.legend(loc="upper left", fontsize=8, frameon=False)
        _style_axis(axis, "Regression: fit continuous value", "Input", "Target")
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
        axis.annotate(
            "Anomaly",
            xy=anomaly,
            xytext=(1.2, 2.1),
            arrowprops={"arrowstyle": "->", "color": "#ef4444"},
            color="#ef4444",
            fontsize=9,
        )
        _style_axis(axis, "K-means: spherical clusters + distance threshold", "Feature 1", "Feature 2")
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
            rotation = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
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
        axis.annotate(
            "Low probability",
            xy=anomaly,
            xytext=(1.25, 2.15),
            arrowprops={"arrowstyle": "->", "color": "#ef4444"},
            color="#ef4444",
            fontsize=9,
        )
        _style_axis(axis, "GMM: elliptical clusters + probability score", "Feature 1", "Feature 2")
        axis.set_xlim(-2.2, 2.6)
        axis.set_ylim(-1.9, 2.5)
        axis.set_aspect("equal", adjustable="box")
        return

    if block_name == "Visual anomaly detection (FOMO-AD)":
        height, width = 150, 220
        xx, yy = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
        concrete = 0.70 + 0.05 * np.sin(6.0 * xx + 4.5 * yy) + 0.06 * rng.normal(size=(height, width))
        concrete = np.clip(concrete, 0.45, 0.92)
        axis.imshow(concrete, cmap="gray", origin="lower", vmin=0, vmax=1)

        crack_x = np.array([32, 50, 69, 92, 114, 139, 164, 188], dtype=float)
        crack_y = np.array([126, 112, 101, 87, 73, 58, 44, 30], dtype=float)
        axis.plot(crack_x, crack_y, color="#111827", lw=3.2, alpha=0.95)
        axis.plot(crack_x, crack_y, color="#374151", lw=1.3, alpha=0.95)

        crack_boxes = [
            (42, 103, 24, 18),
            (90, 76, 26, 18),
            (146, 39, 24, 18),
        ]
        for index, (x_pos, y_pos, box_w, box_h) in enumerate(crack_boxes, start=1):
            center_x = x_pos + box_w // 2
            center_y = y_pos + box_h // 2
            axis.add_patch(Rectangle((x_pos, y_pos), box_w, box_h, fill=False, ec="#dc2626", lw=2.2))
            axis.text(
                x_pos,
                y_pos + box_h + 3,
                f"crack {index}: x={center_x}, y={center_y}",
                fontsize=7.2,
                color="#b91c1c",
                bbox={"fc": "white", "ec": "none", "alpha": 0.75, "pad": 0.5},
            )

        axis.set_xlim(0, width)
        axis.set_ylim(0, height)
        axis.set_title("FOMO-AD: concrete crack localization", fontsize=12, color="#0f172a")
        axis.set_xticks([])
        axis.set_yticks([])
        return

    if block_name == "Image Classification (Transfer Learning)":
        labels = ["gear", "bolt", "bearing", "other"]
        probs = np.array([0.08, 0.12, 0.73, 0.07])
        axis.barh(labels, probs, color=["#94a3b8", "#94a3b8", "#22c55e", "#94a3b8"])
        axis.set_xlim(0, 1)
        _style_axis(axis, "Image classification probabilities", "Probability", "")
        return

    if block_name == "Keyword Spotting (Transfer Learning)":
        time = np.linspace(0, 1, 350)
        waveform = 0.5 * np.sin(2 * np.pi * 6 * time) + 0.25 * np.sin(2 * np.pi * 18 * time)
        waveform += 0.04 * rng.normal(size=time.size)
        axis.plot(time, waveform, color="#0ea5e9", lw=1.3)
        axis.axvspan(0.55, 0.75, color="#22c55e", alpha=0.25, label="keyword window")
        axis.legend(loc="upper left", fontsize=8, frameon=False)
        _style_axis(axis, "Keyword spotting over audio stream", "Time", "Amplitude")
        axis.set_ylim(-1.2, 1.2)
        return

    if block_name == "Object Detection (MobileNetV2 SSD FPN)":
        canvas = np.tile(np.linspace(0.90, 0.76, 220), (150, 1))
        axis.imshow(canvas, cmap="gray", origin="lower", vmin=0, vmax=1)
        axis.add_patch(Rectangle((0, 0), 220, 38, fc="#d6d3d1", ec="none", alpha=0.95))

        apple_center = np.array([62.0, 64.0]) + rng.normal(scale=[0.7, 0.5], size=2)
        axis.add_patch(
            Circle(
                (apple_center[0], apple_center[1]),
                radius=16,
                fc="#ef4444",
                ec="#991b1b",
                lw=1.4,
                alpha=0.95,
            )
        )
        axis.add_patch(Rectangle((apple_center[0] - 1.4, apple_center[1] + 14), 2.8, 8, fc="#7c2d12", ec="none"))
        axis.add_patch(Ellipse((apple_center[0] + 8, apple_center[1] + 18), width=10, height=5, angle=35, fc="#22c55e", ec="#166534", lw=0.8))

        orange_center = np.array([148.0, 58.0]) + rng.normal(scale=[0.8, 0.6], size=2)
        axis.add_patch(
            Circle(
                (orange_center[0], orange_center[1]),
                radius=15,
                fc="#f97316",
                ec="#9a3412",
                lw=1.4,
                alpha=0.95,
            )
        )
        axis.add_patch(Rectangle((orange_center[0] - 1.2, orange_center[1] + 13), 2.4, 6, fc="#7c2d12", ec="none"))
        axis.add_patch(Ellipse((orange_center[0] + 7, orange_center[1] + 16), width=9, height=4.5, angle=28, fc="#22c55e", ec="#166534", lw=0.8))

        detections = [
            (apple_center[0] - 20, apple_center[1] - 20, 40, 44, "apple 0.95"),
            (orange_center[0] - 18, orange_center[1] - 18, 36, 40, "orange 0.93"),
        ]
        for x_pos, y_pos, width, height, label in detections:
            axis.add_patch(Rectangle((x_pos, y_pos), width, height, fill=False, ec="#22c55e", lw=2.2))
            axis.text(
                x_pos,
                y_pos + height + 3,
                label,
                color="#16a34a",
                fontsize=8,
                bbox={"fc": "white", "ec": "none", "alpha": 0.75, "pad": 0.6},
            )

        axis.set_xlim(0, 220)
        axis.set_ylim(0, 150)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title("Object detection: apple and orange", fontsize=12, color="#0f172a")
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
        axis.contourf(
            xx,
            yy,
            regions,
            levels=[-0.5, 0.5, 1.5, 2.5],
            colors=["#dbeafe", "#dcfce7", "#fef3c7"],
            alpha=0.85,
        )
        points = rng.normal(size=(80, 2))
        axis.scatter(points[:, 0], points[:, 1], s=10, color="#0f172a", alpha=0.35)
        _style_axis(axis, "Classical ML: partitioned feature space", "Feature 1", "Feature 2")
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
            axis.add_patch(
                Rectangle(
                    (x_pos, y_pos),
                    width,
                    height,
                    fc="#e2e8f0",
                    ec="#64748b",
                    lw=1.3,
                    transform=axis.transAxes,
                )
            )
            axis.text(
                x_pos + width / 2,
                y_pos + height / 2,
                label,
                ha="center",
                va="center",
                fontsize=9,
                transform=axis.transAxes,
            )
        axis.annotate("", xy=(0.38, 0.51), xytext=(0.29, 0.51), arrowprops={"arrowstyle": "->", "lw": 1.6}, xycoords=axis.transAxes)
        axis.annotate("", xy=(0.74, 0.51), xytext=(0.65, 0.51), arrowprops={"arrowstyle": "->", "lw": 1.6}, xycoords=axis.transAxes)
        return

    axis.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), color="#64748b")
    _style_axis(axis, block_name)


def _figure_to_rgb_array(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    return rgba[..., :3].copy()


def render_reference_concept(catalog_type: str, block_name: str, seed: int = 0) -> np.ndarray:
    plt.close("all")
    fig, axis = plt.subplots(figsize=(8.0, 4.8), dpi=REFERENCE_DPI)

    if catalog_type == "dsp":
        _draw_reference_dsp(axis, block_name, seed=seed)
    else:
        _draw_reference_ml(axis, block_name, seed=seed)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.13)
    image = _figure_to_rgb_array(fig)
    plt.close(fig)
    return image


def render_reference_png(catalog_type: str, block_name: str, output_path: Path, seed: int = 0) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference = render_reference_concept(catalog_type=catalog_type, block_name=block_name, seed=seed)
    imageio.imwrite(output_path, reference)
    return output_path


def generate_reference_scripts(output_dir: Path, specs: Sequence[ConceptSpec] | None = None) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    active_specs = list(specs) if specs is not None else build_concept_specs()

    created: list[Path] = []
    for spec in active_specs:
        script_path = output_dir / f"reference_{spec.catalog_type}_{_slugify_block_name(spec.block_name)}.py"
        png_name = f"reference_{spec.catalog_type}_{_slugify_block_name(spec.block_name)}.png"
        script = (
            "from pathlib import Path\n\n"
            "from concept_validation import render_reference_png\n\n"
            "render_reference_png(\n"
            f"    catalog_type={spec.catalog_type!r},\n"
            f"    block_name={spec.block_name!r},\n"
            f"    output_path=Path({png_name!r}),\n"
            ")\n"
        )
        script_path.write_text(script, encoding="utf-8")
        created.append(script_path)
    return created


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image.astype(np.float32)
    elif image.ndim == 3 and image.shape[2] >= 3:
        gray = (
            0.2989 * image[..., 0].astype(np.float32)
            + 0.5870 * image[..., 1].astype(np.float32)
            + 0.1140 * image[..., 2].astype(np.float32)
        )
    else:
        raise ValueError("Unsupported image shape for grayscale conversion")
    return gray


def _resize_nearest(image: np.ndarray, target_size: int) -> np.ndarray:
    src_h, src_w = image.shape
    y_idx = np.linspace(0, src_h - 1, target_size).astype(np.int32)
    x_idx = np.linspace(0, src_w - 1, target_size).astype(np.int32)
    return image[np.ix_(y_idx, x_idx)]


def _normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    low = float(image.min())
    high = float(image.max())
    if high - low < 1e-8:
        return np.zeros_like(image)
    return (image - low) / (high - low)


def _corrcoef_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_centered = a.ravel().astype(np.float32)
    b_centered = b.ravel().astype(np.float32)
    a_centered -= float(a_centered.mean())
    b_centered -= float(b_centered.mean())
    denom = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a_centered, b_centered) / denom)


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    grad_y, grad_x = np.gradient(image)
    return np.hypot(grad_x, grad_y)


def _histogram_overlap(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    hist_a, _ = np.histogram(a, bins=bins, range=(0.0, 1.0))
    hist_b, _ = np.histogram(b, bins=bins, range=(0.0, 1.0))
    hist_a = hist_a.astype(np.float64)
    hist_b = hist_b.astype(np.float64)
    if hist_a.sum() <= 0 or hist_b.sum() <= 0:
        return 0.0
    hist_a /= hist_a.sum()
    hist_b /= hist_b.sum()
    return float(np.minimum(hist_a, hist_b).sum())


def compute_accuracy_percent(reference_frame: np.ndarray, gif_frame: np.ndarray) -> float:
    ref_gray = _normalize(_resize_nearest(_to_grayscale(reference_frame), REFERENCE_SIZE))
    gif_gray = _normalize(_resize_nearest(_to_grayscale(gif_frame), REFERENCE_SIZE))

    luminance_similarity = max(0.0, _corrcoef_similarity(ref_gray, gif_gray))
    edge_similarity = max(0.0, _corrcoef_similarity(_gradient_magnitude(ref_gray), _gradient_magnitude(gif_gray)))
    histogram_similarity = _histogram_overlap(ref_gray, gif_gray)

    score = 100.0 * (
        0.45 * luminance_similarity
        + 0.35 * edge_similarity
        + 0.20 * histogram_similarity
    )
    return round(float(np.clip(score, 0.0, 100.0)), 2)


def _load_first_gif_frame(gif_path: Path) -> np.ndarray:
    frames = imageio.mimread(gif_path, memtest=False)
    if not frames:
        raise ValueError(f"No frames found in GIF: {gif_path}")
    return np.asarray(frames[0])


def validate_concept(spec: ConceptSpec, img_dir: Path) -> ConceptValidationResult:
    gif_path = img_dir / spec.gif_name
    if not gif_path.exists():
        return ConceptValidationResult(
            catalog_type=spec.catalog_type,
            block_name=spec.block_name,
            gif_path=str(gif_path),
            script_type=spec.script_type,
            accuracy_percent=None,
        )

    reference_frame = render_reference_concept(
        catalog_type=spec.catalog_type,
        block_name=spec.block_name,
        seed=0,
    )
    gif_frame = _load_first_gif_frame(gif_path)
    accuracy = compute_accuracy_percent(reference_frame, gif_frame)

    return ConceptValidationResult(
        catalog_type=spec.catalog_type,
        block_name=spec.block_name,
        gif_path=str(gif_path),
        script_type=spec.script_type,
        accuracy_percent=accuracy,
    )


def validate_all_concepts(img_dir: Path) -> list[ConceptValidationResult]:
    specs = build_concept_specs()
    return [validate_concept(spec=spec, img_dir=img_dir) for spec in specs]


def validate_gif_style(gif_path: Path) -> GifStyleValidationResult:
    if not gif_path.exists():
        return GifStyleValidationResult(
            gif_name=gif_path.name,
            gif_path=str(gif_path),
            header_centered=False,
            watermark_bottom_left=False,
            palette_aligned=False,
            red_poi_present=False,
            requires_red_poi=_requires_red_poi(gif_path.name),
            style_score_percent=0.0,
        )

    frames = _sample_gif_frames(gif_path, max_frames=5)
    palette_rgb = np.stack(
        [_hex_to_rgb(hex_color) for hex_color in STYLE_GUIDE_SPEC["palette_hex"]],
        axis=0,
    )

    header_hits = 0
    watermark_hits = 0
    palette_scores: list[float] = []
    red_ratios: list[float] = []

    for frame in frames:
        frame_rgb = _ensure_rgb(frame)
        background_rgb = _estimate_background_rgb(frame_rgb)

        header_hits += int(_header_centered(frame_rgb, background_rgb))
        watermark_hits += int(_watermark_bottom_left(frame_rgb, background_rgb))
        palette_scores.append(_palette_alignment(frame_rgb, palette_rgb))
        red_ratios.append(_red_poi_ratio(frame_rgb))

    frame_count = max(1, len(frames))
    header_centered = header_hits >= max(1, int(np.ceil(frame_count * 0.5)))
    watermark_bottom_left = watermark_hits >= max(1, int(np.ceil(frame_count * 0.5)))
    palette_alignment_avg = float(np.mean(palette_scores)) if palette_scores else 0.0
    palette_aligned = palette_alignment_avg >= STYLE_CHECK_THRESHOLDS["palette_alignment_min"]

    red_ratio_peak = max(red_ratios) if red_ratios else 0.0
    requires_red_poi = _requires_red_poi(gif_path.name)
    red_poi_present = red_ratio_peak >= STYLE_CHECK_THRESHOLDS["red_poi_min"]

    weighted_score = (
        0.30 * float(header_centered)
        + 0.25 * float(watermark_bottom_left)
        + 0.30 * np.clip(palette_alignment_avg, 0.0, 1.0)
        + 0.15 * (float(red_poi_present) if requires_red_poi else 1.0)
    )
    style_score_percent = round(float(np.clip(weighted_score * 100.0, 0.0, 100.0)), 2)

    return GifStyleValidationResult(
        gif_name=gif_path.name,
        gif_path=str(gif_path),
        header_centered=header_centered,
        watermark_bottom_left=watermark_bottom_left,
        palette_aligned=palette_aligned,
        red_poi_present=red_poi_present,
        requires_red_poi=requires_red_poi,
        style_score_percent=style_score_percent,
    )


def validate_style_for_all_gifs(img_dir: Path) -> list[GifStyleValidationResult]:
    gif_paths = sorted(img_dir.glob("*.gif"))
    return [validate_gif_style(gif_path=gif_path) for gif_path in gif_paths]


def _format_accuracy(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1f}%"


def update_readme_accuracy_columns(readme_path: Path, results: Sequence[ConceptValidationResult]) -> None:
    lines = readme_path.read_text(encoding="utf-8").splitlines()
    accuracy_by_block = {result.block_name: _format_accuracy(result.accuracy_percent) for result in results}

    target_sections = {
        "### DSP Processing Blocks",
        "### ML Learning Blocks",
    }

    rewritten: list[str] = []
    in_target_section = False
    in_target_table = False

    for line in lines:
        stripped = line.strip()

        if stripped in target_sections:
            in_target_section = True
            in_target_table = False
            rewritten.append(line)
            continue

        if in_target_section and stripped.startswith("### ") and stripped not in target_sections:
            in_target_section = False
            in_target_table = False

        if in_target_section and stripped.startswith("| Block |"):
            rewritten.append("| Block | Preview | Concept Accuracy (%) |")
            in_target_table = True
            continue

        if in_target_table and stripped.startswith("|---"):
            rewritten.append("|---|---|---|")
            continue

        if in_target_table and stripped.startswith("|"):
            columns = [column.strip() for column in stripped.strip("|").split("|")]
            if len(columns) >= 2:
                block_name = columns[0]
                preview = columns[1]
                accuracy = accuracy_by_block.get(block_name, "N/A")
                rewritten.append(f"| {block_name} | {preview} | {accuracy} |")
                continue

        if in_target_table and (not stripped.startswith("|")):
            in_target_table = False
            if stripped == "":
                in_target_section = False

        rewritten.append(line)

    readme_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def results_to_json(results: Sequence[ConceptValidationResult]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for result in results:
        payload.append(
            {
                "catalog_type": result.catalog_type,
                "block_name": result.block_name,
                "gif_path": result.gif_path,
                "script_type": result.script_type,
                "accuracy_percent": result.accuracy_percent,
            }
        )
    return payload


def style_results_to_json(results: Sequence[GifStyleValidationResult]) -> dict[str, object]:
    rows = []
    for result in results:
        rows.append(
            {
                "gif_name": result.gif_name,
                "gif_path": result.gif_path,
                "header_centered": result.header_centered,
                "watermark_bottom_left": result.watermark_bottom_left,
                "palette_aligned": result.palette_aligned,
                "red_poi_present": result.red_poi_present,
                "requires_red_poi": result.requires_red_poi,
                "style_score_percent": result.style_score_percent,
            }
        )

    pass_count = sum(
        int(
            row["header_centered"]
            and row["watermark_bottom_left"]
            and row["palette_aligned"]
            and (row["red_poi_present"] if row["requires_red_poi"] else True)
        )
        for row in rows
    )

    summary = {
        "total_gifs": len(rows),
        "style_pass_count": pass_count,
        "style_pass_rate_percent": round(100.0 * pass_count / max(1, len(rows)), 2),
    }
    return {
        "style_guide": STYLE_GUIDE_SPEC,
        "summary": summary,
        "results": rows,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate concept GIFs against reference concept scripts")
    parser.add_argument("--img-dir", default="img", help="Directory containing concept GIFs")
    parser.add_argument("--readme", default="README.md", help="README file path")
    parser.add_argument("--scripts-dir", default="scratch/concept_validation_scripts", help="Output directory for generated reference scripts")
    parser.add_argument("--json-out", default="", help="Optional JSON output path")
    parser.add_argument("--style-check", action="store_true", help="Validate style guide compliance across all GIFs")
    parser.add_argument("--style-json-out", default="", help="Optional JSON output path for style check results")
    parser.add_argument("--generate-scripts", action="store_true", help="Generate one small reference script per concept")
    parser.add_argument("--update-readme", action="store_true", help="Write concept accuracy column into README tables")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    results = validate_all_concepts(img_dir=Path(args.img_dir))
    style_results: list[GifStyleValidationResult] = []

    if args.generate_scripts:
        generate_reference_scripts(output_dir=Path(args.scripts_dir), specs=build_concept_specs())

    if args.update_readme:
        update_readme_accuracy_columns(readme_path=Path(args.readme), results=results)

    if args.json_out:
        payload = results_to_json(results)
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.style_check or args.style_json_out:
        style_results = validate_style_for_all_gifs(img_dir=Path(args.img_dir))
        if args.style_json_out:
            style_payload = style_results_to_json(style_results)
            Path(args.style_json_out).write_text(json.dumps(style_payload, indent=2), encoding="utf-8")

    for result in results:
        print(
            f"{result.catalog_type:3s} | {result.block_name:45s} | {result.script_type:5s} | {_format_accuracy(result.accuracy_percent):>8s}"
        )

    if style_results:
        print("\nStyle guide compliance:")
        for result in style_results:
            required_red = "yes" if result.requires_red_poi else "no"
            print(
                f"style | {result.gif_name:45s} | score={result.style_score_percent:6.2f}% | "
                f"header={result.header_centered} | watermark={result.watermark_bottom_left} | "
                f"palette={result.palette_aligned} | red={result.red_poi_present} (required={required_red})"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
