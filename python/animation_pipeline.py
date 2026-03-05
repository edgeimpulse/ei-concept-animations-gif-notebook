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
CLASSIFICATION_BOUNDARY_SLOPE = -0.65
CLASSIFICATION_BOUNDARY_INTERCEPT = 0.2

PRESET_NAMES: tuple[str, ...] = (
    "dsp_sine_shift",
    "dsp_processing_blocks",
    "dsp_processing_blocks_individual",
    "nn_sigmoid_shift",
    "nn_training_layers",
    "nn_inference_layers",
    "nn_training_vs_on_device_inference",
    "nn_single_neuron",
    "nn_architecture_layers",
    "nn_deep_network",
    "nn_backpropagation_learning",
    "nn_playground_classification",
    "nn_playground_regression",
    "ml_learning_blocks",
    "ml_learning_blocks_individual",
    "embedded_quantization_8bit_vs_float32",
    "sine_bead",
    "gapminder_full",
)

DEFAULT_ALL_PRESETS: tuple[str, ...] = (
    "dsp_sine_shift",
    "dsp_processing_blocks",
    "nn_sigmoid_shift",
    "nn_training_layers",
    "nn_inference_layers",
    "nn_training_vs_on_device_inference",
    "nn_single_neuron",
    "nn_architecture_layers",
    "nn_deep_network",
    "nn_backpropagation_learning",
    "nn_playground_classification",
    "nn_playground_regression",
    "ml_learning_blocks",
    "embedded_quantization_8bit_vs_float32",
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


def _network_node_positions(layer_sizes: Sequence[int]) -> list[list[tuple[float, float]]]:
    x_positions = np.linspace(0.12, 0.88, len(layer_sizes))
    node_positions: list[list[tuple[float, float]]] = []
    for x_pos, layer_size in zip(x_positions, layer_sizes):
        if layer_size <= 1:
            y_positions = np.array([0.50])
        else:
            y_positions = np.linspace(0.18, 0.82, layer_size)
        node_positions.append([(float(x_pos), float(y_pos)) for y_pos in y_positions])
    return node_positions


def _draw_dense_network(
    axis: plt.Axes,
    layer_sizes: Sequence[int],
    active_layer: int | None,
    edge_alpha: float,
    show_backprop: bool,
) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    positions = _network_node_positions(layer_sizes)

    for layer_index in range(len(layer_sizes) - 1):
        for x0, y0 in positions[layer_index]:
            for x1, y1 in positions[layer_index + 1]:
                axis.plot(
                    [x0, x1],
                    [y0, y1],
                    color="#94a3b8",
                    lw=0.85,
                    alpha=edge_alpha,
                    transform=axis.transAxes,
                )

    for layer_index, layer_nodes in enumerate(positions):
        node_color = "#64748b"
        if layer_index == 0:
            node_color = "#2563eb"
        if layer_index == len(positions) - 1:
            node_color = "#16a34a"
        if active_layer is not None and layer_index == active_layer:
            node_color = "#f59e0b"

        for x_pos, y_pos in layer_nodes:
            axis.add_patch(
                Circle(
                    (x_pos, y_pos),
                    radius=0.024,
                    transform=axis.transAxes,
                    fc=node_color,
                    ec="#0f172a",
                    lw=0.7,
                    alpha=0.96,
                )
            )

        axis.text(
            layer_nodes[0][0],
            0.86,
            f"L{layer_index + 1}",
            ha="center",
            fontsize=8,
            color="#334155",
            transform=axis.transAxes,
        )

    if show_backprop:
        for layer_index in range(len(positions) - 1, 0, -1):
            y_pos = 0.07 + 0.02 * (layer_index % 2)
            axis.annotate(
                "",
                xy=(positions[layer_index - 1][0][0], y_pos),
                xytext=(positions[layer_index][0][0], y_pos),
                arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#ef4444", "alpha": 0.88},
                xycoords=axis.transAxes,
            )


@gif.frame
def _nn_training_layers_frame(step: int, total_steps: int, dpi: int = 160) -> None:
    plt.close("all")
    fig = plt.figure(figsize=(6.6, 4.4), dpi=dpi)
    grid = fig.add_gridspec(2, 1, height_ratios=[0.70, 0.30], hspace=0.22)
    network_axis = fig.add_subplot(grid[0, 0])
    loss_axis = fig.add_subplot(grid[1, 0])

    progress = step / max(1, total_steps - 1)
    active_layer = min(3, int(np.floor(progress * 4)))
    edge_alpha = 0.14 + 0.34 * (0.5 + 0.5 * np.sin(2 * np.pi * progress * 2.2))
    epoch = int(np.interp(progress, [0, 1], [1, 40]))

    _draw_dense_network(
        network_axis,
        layer_sizes=(4, 6, 5, 3),
        active_layer=active_layer,
        edge_alpha=edge_alpha,
        show_backprop=True,
    )
    network_axis.set_title(
        f"Training epoch {epoch}: creating and tuning layers",
        fontsize=10,
        color="#0f172a",
        pad=6,
    )

    epochs = np.arange(1, 41)
    loss = 1.45 * np.exp(-epochs / 12.5) + 0.08
    shown = max(2, int(np.ceil(progress * len(epochs))))
    loss_axis.plot(epochs, loss, color="#cbd5e1", lw=1.5)
    loss_axis.plot(epochs[:shown], loss[:shown], color="#2563eb", lw=2.2)
    loss_axis.scatter([epochs[shown - 1]], [loss[shown - 1]], color="#1d4ed8", s=24)
    loss_axis.set_xlim(1, 40)
    loss_axis.set_ylim(0, 1.6)
    loss_axis.set_title("Loss during training", fontsize=10, color="#0f172a")
    loss_axis.set_xlabel("Epoch", fontsize=9)
    loss_axis.set_ylabel("Loss", fontsize=9)
    loss_axis.grid(alpha=0.25)

    fig.suptitle("NN Training: layer creation + weight updates", y=0.98, fontsize=13, color="#0f172a")


def render_nn_training_layers(
    output_path: Path,
    frame_count: int = 24,
    hold_last: int = 6,
    duration_ms: int = 95,
    optimize: bool = True,
) -> Path:
    frames = [_nn_training_layers_frame(step, frame_count) for step in range(frame_count)]
    frames.extend(
        [_nn_training_layers_frame(frame_count - 1, frame_count) for _ in range(max(0, hold_last))]
    )
    gif.save(frames, str(output_path), duration=_slow_duration_ms(duration_ms))
    maybe_optimize_gif(output_path, optimize=optimize)
    return output_path


@gif.frame
def _nn_inference_layers_frame(step: int, total_steps: int, dpi: int = 160) -> None:
    plt.close("all")
    fig = plt.figure(figsize=(6.8, 4.4), dpi=dpi)
    grid = fig.add_gridspec(1, 2, width_ratios=[0.68, 0.32], wspace=0.20)
    network_axis = fig.add_subplot(grid[0, 0])
    output_axis = fig.add_subplot(grid[0, 1])

    progress = step / max(1, total_steps - 1)
    active_layer = min(3, int(np.floor(progress * 4)))

    _draw_dense_network(
        network_axis,
        layer_sizes=(4, 6, 5, 3),
        active_layer=active_layer,
        edge_alpha=0.18,
        show_backprop=False,
    )
    network_axis.text(
        0.02,
        0.97,
        f"Inference sample #{step + 1}: forward pass only",
        ha="left",
        va="top",
        fontsize=10,
        color="#0f172a",
        transform=network_axis.transAxes,
    )

    logits = np.array(
        [
            1.2 + 0.8 * np.sin(0.33 * step),
            0.7 + 0.9 * np.cos(0.22 * step),
            0.5 + 0.6 * np.sin(0.41 * step + 0.6),
        ]
    )
    shifted = logits - logits.max()
    probs = np.exp(shifted) / np.exp(shifted).sum()
    labels = ["class A", "class B", "class C"]
    top_index = int(np.argmax(probs))
    colors = ["#94a3b8", "#94a3b8", "#94a3b8"]
    colors[top_index] = "#16a34a"

    output_axis.barh(labels, probs, color=colors)
    output_axis.set_xlim(0, 1)
    output_axis.set_xlabel("Probability", fontsize=9)
    output_axis.set_title("Output", fontsize=10, color="#0f172a")
    output_axis.grid(alpha=0.25, axis="x")
    output_axis.text(
        0.03,
        0.03,
        f"Prediction: {labels[top_index]}",
        transform=output_axis.transAxes,
        fontsize=9,
        color="#166534",
    )

    fig.suptitle("NN Inference: fixed layers + forward execution", y=0.98, fontsize=13, color="#0f172a")


def render_nn_inference_layers(
    output_path: Path,
    frame_count: int = 24,
    hold_last: int = 6,
    duration_ms: int = 95,
    optimize: bool = True,
) -> Path:
    frames = [_nn_inference_layers_frame(step, frame_count) for step in range(frame_count)]
    frames.extend(
        [_nn_inference_layers_frame(frame_count - 1, frame_count) for _ in range(max(0, hold_last))]
    )
    gif.save(frames, str(output_path), duration=_slow_duration_ms(duration_ms))
    maybe_optimize_gif(output_path, optimize=optimize)
    return output_path


@gif.frame
def _nn_training_vs_on_device_inference_frame(step: int, total_steps: int, dpi: int = 160) -> None:
    plt.close("all")
    fig = plt.figure(figsize=(8.0, 4.4), dpi=dpi)
    grid = fig.add_gridspec(1, 2, width_ratios=[0.5, 0.5], wspace=0.08)
    train_axis = fig.add_subplot(grid[0, 0])
    infer_axis = fig.add_subplot(grid[0, 1])

    progress = step / max(1, total_steps - 1)
    active_layer = min(3, int(np.floor(progress * 4)))
    epoch = int(np.interp(progress, [0, 1], [1, 50]))

    _draw_dense_network(
        train_axis,
        layer_sizes=(4, 7, 5, 3),
        active_layer=active_layer,
        edge_alpha=0.16 + 0.28 * (0.5 + 0.5 * np.sin(step * 0.45)),
        show_backprop=True,
    )
    train_axis.text(0.02, 0.98, "Cloud training", va="top", fontsize=11, color="#1e3a8a", transform=train_axis.transAxes)
    train_axis.text(0.02, 0.03, f"Epoch {epoch} | float32 optimization", fontsize=8.8, color="#334155", transform=train_axis.transAxes)

    _draw_dense_network(
        infer_axis,
        layer_sizes=(4, 5, 4, 3),
        active_layer=active_layer,
        edge_alpha=0.18,
        show_backprop=False,
    )
    infer_axis.add_patch(
        Rectangle(
            (0.28, 0.02),
            0.44,
            0.09,
            transform=infer_axis.transAxes,
            fc="#e2e8f0",
            ec="#64748b",
            lw=1.1,
        )
    )
    infer_axis.text(0.50, 0.065, "On-device MCU", ha="center", va="center", fontsize=8.5, color="#334155", transform=infer_axis.transAxes)
    infer_axis.text(0.02, 0.98, "On-device inference", va="top", fontsize=11, color="#166534", transform=infer_axis.transAxes)
    infer_axis.text(0.02, 0.03, "Frozen weights | low latency", fontsize=8.8, color="#334155", transform=infer_axis.transAxes)

    fig.suptitle("Training vs on-device inference", y=0.98, fontsize=13, color="#0f172a")


def render_nn_training_vs_on_device_inference(
    output_path: Path,
    frame_count: int = 24,
    hold_last: int = 6,
    duration_ms: int = 95,
    optimize: bool = True,
) -> Path:
    frames = [
        _nn_training_vs_on_device_inference_frame(step, frame_count)
        for step in range(frame_count)
    ]
    frames.extend(
        [
            _nn_training_vs_on_device_inference_frame(frame_count - 1, frame_count)
            for _ in range(max(0, hold_last))
        ]
    )
    gif.save(frames, str(output_path), duration=_slow_duration_ms(duration_ms))
    maybe_optimize_gif(output_path, optimize=optimize)
    return output_path


@gif.frame
def _embedded_quantization_8bit_vs_float32_frame(
    step: int, total_steps: int, dpi: int = 160
) -> None:
    plt.close("all")
    fig = plt.figure(figsize=(6.8, 4.4), dpi=dpi)
    grid = fig.add_gridspec(2, 1, height_ratios=[0.68, 0.32], hspace=0.22)
    weights_axis = fig.add_subplot(grid[0, 0])
    metrics_axis = fig.add_subplot(grid[1, 0])

    base_weights = np.array([-0.82, -0.55, -0.22, 0.03, 0.21, 0.43, 0.66, 0.87])
    jitter = 0.03 * np.sin(np.linspace(0, 2 * np.pi, base_weights.size) + 0.38 * step)
    float_weights = np.clip(base_weights + jitter, -1.0, 1.0)
    int8_weights = np.round(float_weights * 127).astype(np.int8)
    dequant_weights = int8_weights.astype(np.float32) / 127.0

    x_values = np.arange(base_weights.size)
    width = 0.38
    weights_axis.bar(
        x_values - width / 2,
        float_weights,
        width,
        color="#0ea5e9",
        alpha=0.9,
        label="float32",
    )
    weights_axis.bar(
        x_values + width / 2,
        dequant_weights,
        width,
        color="#f59e0b",
        alpha=0.85,
        label="int8 (dequantized)",
    )
    weights_axis.set_ylim(-1.05, 1.05)
    weights_axis.set_xticks(x_values)
    weights_axis.set_xlabel("Weight index", fontsize=9)
    weights_axis.set_ylabel("Value", fontsize=9)
    weights_axis.grid(alpha=0.25)
    weights_axis.legend(loc="lower right", fontsize=8, frameon=False)
    weights_axis.set_title("Quantized weights preserve the shape of float32 weights", fontsize=10, color="#0f172a")

    float_model_size = 4.0
    int8_model_size = 1.0
    float_latency = 1.0
    int8_latency = 0.62 + 0.05 * np.sin(0.5 * step)
    labels = ["Model size", "Inference latency"]
    y_values = np.arange(len(labels))
    metric_height = 0.34

    metrics_axis.barh(y_values - metric_height / 2, [float_model_size, float_latency], metric_height, color="#0ea5e9", label="float32")
    metrics_axis.barh(y_values + metric_height / 2, [int8_model_size, int8_latency], metric_height, color="#f59e0b", label="int8")
    metrics_axis.set_xlim(0, 4.4)
    metrics_axis.set_yticks(y_values)
    metrics_axis.set_yticklabels(labels)
    metrics_axis.grid(alpha=0.25, axis="x")
    metrics_axis.set_xlabel("Relative cost", fontsize=9)
    metrics_axis.legend(loc="lower right", fontsize=8, frameon=False)

    reduction_percent = int(round((1.0 - int8_model_size / float_model_size) * 100))
    metrics_axis.text(
        0.02,
        0.05,
        f"Memory reduction: {reduction_percent}% (8-bit vs float32)",
        transform=metrics_axis.transAxes,
        fontsize=8.8,
        color="#334155",
    )

    fig.suptitle("Embedded: quantization (8-bit vs float32)", y=0.98, fontsize=13, color="#0f172a")


def render_embedded_quantization_8bit_vs_float32(
    output_path: Path,
    frame_count: int = 22,
    hold_last: int = 6,
    duration_ms: int = 95,
    optimize: bool = True,
) -> Path:
    frames = [
        _embedded_quantization_8bit_vs_float32_frame(step, frame_count)
        for step in range(frame_count)
    ]
    frames.extend(
        [
            _embedded_quantization_8bit_vs_float32_frame(frame_count - 1, frame_count)
            for _ in range(max(0, hold_last))
        ]
    )
    gif.save(frames, str(output_path), duration=_slow_duration_ms(duration_ms))
    maybe_optimize_gif(output_path, optimize=optimize)
    return output_path


@gif.frame
def _nn_single_neuron_frame(step: int, total_steps: int, dpi: int = 160) -> None:
    plt.close("all")
    fig = plt.figure(figsize=(6.5, 6.5), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    progress = step / max(1, total_steps - 1)

    # Static geometry
    inputs = [(0.12, 0.72), (0.12, 0.50), (0.12, 0.28)]
    input_labels = ["x₁", "x₂", "x₃"]
    neuron_radius = 0.08
    neuron_x, neuron_y = 0.50, 0.50

    # Animated values (only these change)
    w1 = 0.80 + 0.28 * np.sin(2.0 * np.pi * progress)
    w2 = -0.50 + 0.22 * np.cos(2.4 * np.pi * progress + 0.35)
    w3 = 0.30 + 0.18 * np.sin(1.8 * np.pi * progress + 1.20)
    bias = 0.10 + 0.12 * np.cos(2.2 * np.pi * progress + 0.80)

    learning_rate = 0.030 + 0.012 * (0.5 + 0.5 * np.sin(2.0 * np.pi * progress * 0.9))
    regularization_rate = 0.002 + 0.018 * (0.5 + 0.5 * np.cos(2.0 * np.pi * progress * 0.8 + 0.50))
    epochs = int(np.interp(progress, [0.0, 1.0], [1, 50]))
    activation_options = ("Tanh", "ReLU", "Sigmoid")
    activation_name = activation_options[min(len(activation_options) - 1, int(progress * len(activation_options)))]

    # Draw static input nodes and labels
    for (x_pos, y_pos), label in zip(inputs, input_labels):
        ax.add_patch(
            Circle((x_pos, y_pos), neuron_radius * 0.40, fc="#2563eb", ec="#0f172a", lw=0.8, alpha=0.96, transform=ax.transAxes)
        )
        ax.text(x_pos - 0.06, y_pos, label, ha="right", va="center", fontsize=11, color="#0f172a", transform=ax.transAxes)

    # Draw static weighted connections, labels animate numerically
    current_weights = (w1, w2, w3)
    for (x_pos, y_pos), weight_value in zip(inputs, current_weights):
        ax.plot([x_pos, neuron_x], [y_pos, neuron_y], color="#64748b", lw=1.9, alpha=0.92, transform=ax.transAxes)
        mid_x = (x_pos + neuron_x) / 2
        mid_y = (y_pos + neuron_y) / 2 + 0.03
        weight_color = "#16a34a" if weight_value >= 0 else "#dc2626"
        ax.text(
            mid_x,
            mid_y,
            f"w={weight_value:+.2f}",
            ha="center",
            fontsize=8,
            color=weight_color,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.18", fc="#ffffff", ec="#cbd5e1", alpha=0.9),
        )

    # Draw static neuron body
    ax.add_patch(
        Circle((neuron_x, neuron_y), neuron_radius, fc="#f59e0b", ec="#0f172a", lw=1.4, alpha=0.96, transform=ax.transAxes)
    )
    ax.text(neuron_x, neuron_y + 0.02, "Σ", ha="center", va="center", fontsize=20, color="#ffffff", weight="bold", transform=ax.transAxes)
    ax.text(neuron_x, neuron_y - 0.03, f"+b={bias:+.2f}", ha="center", va="center", fontsize=7.8, color="#ffffff", transform=ax.transAxes)

    # Draw static activation block and output arrow
    ax.annotate(
        "",
        xy=(0.72, neuron_y),
        xytext=(neuron_x + 0.09, neuron_y),
        arrowprops={"arrowstyle": "->", "lw": 2.0, "color": "#64748b", "alpha": 0.9},
        xycoords=ax.transAxes,
    )
    ax.add_patch(Rectangle((0.72, neuron_y - 0.05), 0.12, 0.10, fc="#8b5cf6", ec="#0f172a", lw=1.2, alpha=0.95, transform=ax.transAxes))
    ax.text(0.78, neuron_y, "f(x)", ha="center", va="center", fontsize=10, color="#ffffff", weight="bold", transform=ax.transAxes)
    ax.text(0.78, neuron_y - 0.08, activation_name, ha="center", va="top", fontsize=7, color="#334155", transform=ax.transAxes)

    ax.annotate(
        "",
        xy=(0.92, neuron_y),
        xytext=(0.85, neuron_y),
        arrowprops={"arrowstyle": "->", "lw": 2.0, "color": "#64748b", "alpha": 0.9},
        xycoords=ax.transAxes,
    )
    ax.text(0.94, neuron_y, "y", ha="left", va="center", fontsize=11, color="#0f172a", transform=ax.transAxes)

    # Hyperparameter panel (values animate)
    panel_x, panel_y, panel_w, panel_h = 0.64, 0.17, 0.31, 0.24
    ax.add_patch(Rectangle((panel_x, panel_y), panel_w, panel_h, fc="#f8fafc", ec="#cbd5e1", lw=1.1, transform=ax.transAxes))
    ax.text(panel_x + 0.02, panel_y + panel_h - 0.04, "Hyperparameters", fontsize=8.6, color="#0f172a", weight="bold", transform=ax.transAxes)
    ax.text(panel_x + 0.02, panel_y + panel_h - 0.085, f"Epoch: {epochs:02d}", fontsize=8, color="#334155", transform=ax.transAxes)
    ax.text(panel_x + 0.02, panel_y + panel_h - 0.125, f"Learning rate: {learning_rate:.3f}", fontsize=8, color="#334155", transform=ax.transAxes)
    ax.text(panel_x + 0.02, panel_y + panel_h - 0.165, f"Reg rate: {regularization_rate:.3f}", fontsize=8, color="#334155", transform=ax.transAxes)
    ax.text(panel_x + 0.02, panel_y + panel_h - 0.205, f"Activation: {activation_name}", fontsize=8, color="#334155", transform=ax.transAxes)

    # Title and equation stay visible in all frames
    ax.text(0.50, 0.95, "Single Neuron: weights + hyperparameters update", ha="center", va="top", fontsize=12, color="#0f172a", weight="bold", transform=ax.transAxes)
    ax.text(
        0.50,
        0.11,
        f"y = f({w1:+.2f}x₁ {w2:+.2f}x₂ {w3:+.2f}x₃ {bias:+.2f})",
        ha="center",
        va="center",
        fontsize=9.8,
        color="#334155",
        style="italic",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.45", fc="#f1f5f9", ec="#cbd5e1", alpha=0.95),
    )


def render_nn_single_neuron(
    output_path: Path,
    frame_count: int = 28,
    hold_last: int = 8,
    duration_ms: int = 100,
    optimize: bool = True,
) -> Path:
    frames = [_nn_single_neuron_frame(step, frame_count) for step in range(frame_count)]
    frames.extend([_nn_single_neuron_frame(frame_count - 1, frame_count) for _ in range(max(0, hold_last))])
    gif.save(frames, str(output_path), duration=_slow_duration_ms(duration_ms))
    maybe_optimize_gif(output_path, optimize=optimize)
    return output_path


@gif.frame
def _nn_architecture_layers_frame(step: int, total_steps: int, dpi: int = 160) -> None:
    plt.close("all")
    fig = plt.figure(figsize=(6.5, 6.5), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis("off")

    progress = step / max(1, total_steps - 1)
    
    # Define three layers: input, hidden, output
    layers = [(0.20, 4, "#2563eb", "Input\nLayer"), 
              (0.50, 6, "#f59e0b", "Hidden\nLayer"), 
              (0.80, 3, "#16a34a", "Output\nLayer")]
    
    # Flow animation - which layer is currently processing
    flow_layer = int((progress * 2.5) % 3.5)
    
    # Draw connections between layers with flow
    for layer_idx in range(len(layers) - 1):
        if progress > 0.2 + layer_idx * 0.25:
            alpha = min(1.0, (progress - (0.2 + layer_idx * 0.25)) * 3.0)
            x0, size0, _, _ = layers[layer_idx]
            x1, size1, _, _ = layers[layer_idx + 1]
            
            y_positions_0 = np.linspace(0.25, 0.75, size0)
            y_positions_1 = np.linspace(0.25, 0.75, size1)
            
            # Highlight connections when data flows through
            is_flowing = (flow_layer == layer_idx and progress > 0.5)
            conn_color = "#10b981" if is_flowing else "#cbd5e1"
            conn_lw = 1.2 if is_flowing else 0.7
            
            for y0 in y_positions_0:
                for y1 in y_positions_1:
                    ax.plot([x0, x1], [y0, y1], color=conn_color, lw=conn_lw, alpha=alpha * 0.5, transform=ax.transAxes)
    
    # Draw nodes for each layer with flow highlight
    for layer_idx, (x_pos, num_nodes, color, label) in enumerate(layers):
        if progress > layer_idx * 0.25:
            alpha = min(1.0, (progress - layer_idx * 0.25) * 2.5)
            y_positions = np.linspace(0.25, 0.75, num_nodes)
            
            # Highlight active layer
            is_active = (flow_layer == layer_idx and progress > 0.5)
            node_color = "#10b981" if is_active else color
            
            for y_pos in y_positions:
                ax.add_patch(Circle((x_pos, y_pos), 0.028, fc=node_color, ec="#0f172a", 
                                   lw=0.8, alpha=alpha, transform=ax.transAxes))
            
            # Layer label
            ax.text(x_pos, 0.10, label, ha="center", va="center", fontsize=10, 
                   color=color, alpha=alpha, weight="bold", transform=ax.transAxes)
    
    # Title
    ax.text(0.50, 0.94, "Neural Network Architecture: Layers", 
           ha="center", va="top", fontsize=12, color="#0f172a", weight="bold", transform=ax.transAxes)
    
    # Description
    if progress > 0.7:
        text_alpha = min(1.0, (progress - 0.7) * 2.0)
        ax.text(0.50, 0.02, "Data flows from input → hidden → output layers", 
               ha="center", va="bottom", fontsize=9, color="#334155", 
               style="italic", alpha=text_alpha, transform=ax.transAxes)


def render_nn_architecture_layers(
    output_path: Path,
    frame_count: int = 26,
    hold_last: int = 8,
    duration_ms: int = 100,
    optimize: bool = True,
) -> Path:
    frames = [_nn_architecture_layers_frame(step, frame_count) for step in range(frame_count)]
    frames.extend([_nn_architecture_layers_frame(frame_count - 1, frame_count) for _ in range(max(0, hold_last))])
    gif.save(frames, str(output_path), duration=_slow_duration_ms(duration_ms))
    maybe_optimize_gif(output_path, optimize=optimize)
    return output_path


@gif.frame
def _nn_deep_network_frame(step: int, total_steps: int, dpi: int = 160) -> None:
    plt.close("all")
    fig = plt.figure(figsize=(7.5, 7.5), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis("off")

    progress = step / max(1, total_steps - 1)
    
    # Deep network with multiple hidden layers
    layer_sizes = [5, 8, 10, 8, 6, 3]
    colors = ["#2563eb", "#f59e0b", "#f59e0b", "#f59e0b", "#f59e0b", "#16a34a"]
    labels = ["Input", "Hidden 1", "Hidden 2", "Hidden 3", "Hidden 4", "Output"]
    
    positions = _network_node_positions(layer_sizes)
    
    # Flow animation - which layer is processing
    flow_layer = int((progress * 2.5) % len(layer_sizes))
    
    # Draw connections with progressive reveal and flow
    for layer_idx in range(len(layer_sizes) - 1):
        if progress > 0.15 + layer_idx * 0.12:
            alpha = min(1.0, (progress - (0.15 + layer_idx * 0.12)) * 5.0)
            # Highlight connections when flow passes through
            is_flowing = (flow_layer == layer_idx and progress > 0.6)
            conn_color = "#10b981" if is_flowing else "#cbd5e1"
            conn_lw = 0.9 if is_flowing else 0.5
            for x0, y0 in positions[layer_idx]:
                for x1, y1 in positions[layer_idx + 1]:
                    ax.plot([x0, x1], [y0, y1], color=conn_color, lw=conn_lw, 
                           alpha=alpha * 0.4, transform=ax.transAxes)
    
    # Draw nodes with flow highlight
    for layer_idx, (layer_nodes, color, label) in enumerate(zip(positions, colors, labels)):
        if progress > layer_idx * 0.12:
            alpha = min(1.0, (progress - layer_idx * 0.12) * 3.0)
            # Highlight active layer during flow
            is_active = (flow_layer == layer_idx and progress > 0.6)
            node_color = "#10b981" if is_active else color
            for x_pos, y_pos in layer_nodes:
                ax.add_patch(Circle((x_pos, y_pos), 0.018, fc=node_color, ec="#0f172a", 
                                   lw=0.6, alpha=alpha, transform=ax.transAxes))
            
            # Layer label
            ax.text(layer_nodes[0][0], 0.08, label, ha="center", va="center", 
                   fontsize=8, color="#334155", alpha=alpha, transform=ax.transAxes)
    
    # Title
    ax.text(0.50, 0.96, "Deep Neural Network: Multiple Hidden Layers", 
           ha="center", va="top", fontsize=12, color="#0f172a", weight="bold", transform=ax.transAxes)
    
    # Description with depth info
    if progress > 0.75:
        text_alpha = min(1.0, (progress - 0.75) * 3.0)
        ax.text(0.50, 0.02, f"Deep networks have many layers (6 layers shown, 4 hidden)", 
               ha="center", va="bottom", fontsize=9, color="#334155", 
               style="italic", alpha=text_alpha, transform=ax.transAxes)


def render_nn_deep_network(
    output_path: Path,
    frame_count: int = 30,
    hold_last: int = 8,
    duration_ms: int = 95,
    optimize: bool = True,
) -> Path:
    effective_steps = max(2, frame_count)
    static_step = effective_steps - 1
    frames = [_nn_deep_network_frame(static_step, effective_steps)]
    gif.save(frames, str(output_path), duration=_slow_duration_ms(duration_ms))
    maybe_optimize_gif(output_path, optimize=optimize)
    return output_path


@gif.frame
def _nn_backpropagation_learning_frame(step: int, total_steps: int, dpi: int = 160) -> None:
    plt.close("all")
    fig = plt.figure(figsize=(8.0, 8.0), dpi=dpi)
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.42, 0.42, 0.16],
        width_ratios=[0.50, 0.50],
        hspace=0.15,
        wspace=0.12,
    )
    forward_axis = fig.add_subplot(grid[0, 0])
    backward_axis = fig.add_subplot(grid[0, 1])
    combined_axis = fig.add_subplot(grid[1, :])
    info_axis = fig.add_subplot(grid[2, :])

    progress = step / max(1, total_steps - 1)
    is_forward_phase = progress < 0.5
    phase_progress = progress / 0.5 if is_forward_phase else (progress - 0.5) / 0.5

    layer_sizes = [4, 6, 4, 3]
    positions = _network_node_positions(layer_sizes)

    # === FORWARD PASS (Top Left) ===
    forward_axis.set_xlim(0, 1)
    forward_axis.set_ylim(0, 1)
    forward_axis.set_aspect("equal")
    forward_axis.axis("off")
    forward_axis.set_title("Forward Pass: Input → Output", fontsize=10, color="#047857", weight="bold", pad=8)

    for layer_idx in range(len(layer_sizes) - 1):
        for x0, y0 in positions[layer_idx]:
            for x1, y1 in positions[layer_idx + 1]:
                forward_axis.plot([x0, x1], [y0, y1], color="#e5e7eb", lw=0.6, alpha=0.5, transform=forward_axis.transAxes)

    active_forward_layer = min(len(layer_sizes) - 1, int(np.floor(phase_progress * len(layer_sizes)))) if is_forward_phase else len(layer_sizes) - 1

    for layer_idx in range(active_forward_layer):
        for x0, y0 in positions[layer_idx]:
            for x1, y1 in positions[layer_idx + 1]:
                forward_axis.plot([x0, x1], [y0, y1], color="#10b981", lw=1.2, alpha=0.6, transform=forward_axis.transAxes)

    for layer_idx in range(len(layer_sizes)):
        for x_pos, y_pos in positions[layer_idx]:
            if layer_idx < active_forward_layer:
                node_color = "#10b981"
            elif layer_idx == active_forward_layer and is_forward_phase:
                node_color = "#34d399"
            else:
                node_color = "#d1d5db"
            forward_axis.add_patch(Circle((x_pos, y_pos), 0.022, fc=node_color, ec="#0f172a", lw=0.7, alpha=0.95, transform=forward_axis.transAxes))

    # === BACKWARD PASS (Top Right) ===
    backward_axis.set_xlim(0, 1)
    backward_axis.set_ylim(0, 1)
    backward_axis.set_aspect("equal")
    backward_axis.axis("off")
    backward_axis.set_title("Backpropagation: Output → Input", fontsize=10, color="#dc2626", weight="bold", pad=8)

    for layer_idx in range(len(layer_sizes) - 1):
        for x0, y0 in positions[layer_idx]:
            for x1, y1 in positions[layer_idx + 1]:
                backward_axis.plot([x0, x1], [y0, y1], color="#e5e7eb", lw=0.6, alpha=0.5, transform=backward_axis.transAxes)

    if is_forward_phase:
        active_back_layer = len(layer_sizes) - 1
    else:
        reverse_step = min(len(layer_sizes) - 1, int(np.floor(phase_progress * len(layer_sizes))))
        active_back_layer = len(layer_sizes) - 1 - reverse_step
        for layer_idx in range(len(layer_sizes) - 1, max(active_back_layer, 0), -1):
            for x0, y0 in positions[layer_idx - 1]:
                for x1, y1 in positions[layer_idx]:
                    backward_axis.plot([x0, x1], [y0, y1], color="#ef4444", lw=1.2, alpha=0.6, transform=backward_axis.transAxes)

    for layer_idx in range(len(layer_sizes)):
        for x_pos, y_pos in positions[layer_idx]:
            if not is_forward_phase and layer_idx > active_back_layer:
                node_color = "#ef4444"
            elif not is_forward_phase and layer_idx == active_back_layer:
                node_color = "#f87171"
            else:
                node_color = "#d1d5db"
            backward_axis.add_patch(Circle((x_pos, y_pos), 0.022, fc=node_color, ec="#0f172a", lw=0.7, alpha=0.95, transform=backward_axis.transAxes))

    # === COMBINED VIEW (Middle) ===
    combined_axis.set_xlim(0, 1)
    combined_axis.set_ylim(0, 1)
    combined_axis.set_aspect("equal")
    combined_axis.axis("off")
    combined_axis.set_title("Single Epoch Cycle", fontsize=10, color="#0f172a", weight="bold", pad=8)

    for layer_idx in range(len(layer_sizes) - 1):
        for x0, y0 in positions[layer_idx]:
            for x1, y1 in positions[layer_idx + 1]:
                combined_axis.plot([x0, x1], [y0, y1], color="#cbd5e1", lw=0.7, alpha=0.4, transform=combined_axis.transAxes)

    if is_forward_phase:
        active_layer = min(len(layer_sizes) - 1, int(np.floor(phase_progress * len(layer_sizes))))
        for layer_idx in range(len(layer_sizes)):
            for x_pos, y_pos in positions[layer_idx]:
                node_color = "#10b981" if layer_idx <= active_layer else "#94a3b8"
                combined_axis.add_patch(Circle((x_pos, y_pos), 0.020, fc=node_color, ec="#0f172a", lw=0.6, alpha=0.95, transform=combined_axis.transAxes))
        combined_axis.annotate("", xy=(0.88, 0.50), xytext=(0.12, 0.50), arrowprops={"arrowstyle": "->", "lw": 2.5, "color": "#10b981", "alpha": 0.7}, xycoords=combined_axis.transAxes)
    else:
        reverse_step = min(len(layer_sizes) - 1, int(np.floor(phase_progress * len(layer_sizes))))
        active_back_layer = len(layer_sizes) - 1 - reverse_step
        for layer_idx in range(len(layer_sizes)):
            for x_pos, y_pos in positions[layer_idx]:
                node_color = "#ef4444" if layer_idx >= active_back_layer else "#94a3b8"
                combined_axis.add_patch(Circle((x_pos, y_pos), 0.020, fc=node_color, ec="#0f172a", lw=0.6, alpha=0.95, transform=combined_axis.transAxes))
        combined_axis.annotate("", xy=(0.12, 0.50), xytext=(0.88, 0.50), arrowprops={"arrowstyle": "->", "lw": 2.5, "color": "#ef4444", "alpha": 0.7}, xycoords=combined_axis.transAxes)

    # === INFO PANEL (Bottom) ===
    info_axis.set_xlim(0, 1)
    info_axis.set_ylim(0, 1)
    info_axis.axis("off")

    start_loss = 1.42
    end_loss = 1.12
    loss = start_loss if is_forward_phase else start_loss - (start_loss - end_loss) * phase_progress

    if is_forward_phase:
        phase_text = "Epoch 1/1 — Forward pass (prediction)"
        phase_color = "#047857"
    else:
        phase_text = "Epoch 1/1 — Backpropagation (weight update)"
        phase_color = "#dc2626"

    info_axis.text(
        0.5,
        0.62,
        f"Loss in this epoch: {loss:.3f}",
        ha="center",
        va="center",
        fontsize=10,
        color="#334155",
        weight="bold",
        transform=info_axis.transAxes,
    )
    info_axis.text(
        0.5,
        0.22,
        phase_text,
        ha="center",
        va="center",
        fontsize=9,
        color=phase_color,
        transform=info_axis.transAxes,
        bbox=dict(boxstyle="round,pad=0.6", fc="#f8fafc", ec=phase_color, lw=1.5),
    )

    fig.suptitle("Neural Network Learning: One Epoch (Forward + Backprop)", y=0.98, fontsize=13, color="#0f172a", weight="bold")


def render_nn_backpropagation_learning(
    output_path: Path,
    frame_count: int = 48,
    hold_last: int = 12,
    duration_ms: int = 120,
    optimize: bool = True,
) -> Path:
    frames = [_nn_backpropagation_learning_frame(step, frame_count) for step in range(frame_count)]
    frames.extend([_nn_backpropagation_learning_frame(frame_count - 1, frame_count) for _ in range(max(0, hold_last))])
    gif.save(frames, str(output_path), duration=_slow_duration_ms(duration_ms))
    maybe_optimize_gif(output_path, optimize=optimize)
    return output_path


def _draw_nn_playground_controls(
    axis: plt.Axes,
    problem_type: str,
    epoch: int,
    train_loss: float,
    test_loss: float,
) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.set_facecolor("#f8fafc")

    def draw_panel(y: float, height: float, title: str, lines: list[str]) -> None:
        axis.add_patch(
            Rectangle((0.04, y), 0.92, height, fc="#ffffff", ec="#cbd5e1", lw=1.0, transform=axis.transAxes)
        )
        axis.text(0.08, y + height - 0.055, title, fontsize=8.6, color="#0f172a", weight="bold", transform=axis.transAxes)
        for index, line in enumerate(lines):
            axis.text(
                0.08,
                y + height - 0.105 - index * 0.048,
                line,
                fontsize=7.8,
                color="#334155",
                transform=axis.transAxes,
            )

    draw_panel(
        0.70,
        0.26,
        "DATA",
        [
            "Dataset: circles",
            "Train/Test: 50%",
            "Noise: 0",
            "Batch size: 10",
        ],
    )
    draw_panel(
        0.40,
        0.26,
        "FEATURES",
        [
            "x₁, x₂",
            "x₁·x₂",
            "sin(x₁), sin(x₂)",
        ],
    )
    draw_panel(
        0.20,
        0.16,
        "MODEL",
        [
            "Learning rate: 0.03",
            "Activation: Tanh",
            f"Problem: {problem_type}",
        ],
    )
    draw_panel(
        0.03,
        0.14,
        "STATUS",
        [
            f"Epoch: {epoch:06d}",
            f"Test loss: {test_loss:.3f}",
            f"Train loss: {train_loss:.3f}",
        ],
    )


def _draw_nn_playground_network(axis: plt.Axes, progress: float, seed: int = 0) -> None:
    layer_sizes = (2, 6, 4, 2, 1)
    positions = _network_node_positions(layer_sizes)
    rng = np.random.default_rng(210 + seed)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_aspect("equal")
    axis.axis("off")

    for layer_index in range(len(layer_sizes) - 1):
        weights = rng.normal(loc=0.0, scale=1.0, size=(layer_sizes[layer_index], layer_sizes[layer_index + 1]))
        max_abs = float(np.max(np.abs(weights))) + 1e-6
        for node_index, (x0, y0) in enumerate(positions[layer_index]):
            for next_index, (x1, y1) in enumerate(positions[layer_index + 1]):
                normalized = weights[node_index, next_index] / max_abs
                scaled = normalized * (0.35 + 0.65 * progress)
                edge_color = "#10b981" if scaled >= 0 else "#ef4444"
                axis.plot(
                    [x0, x1],
                    [y0, y1],
                    color=edge_color,
                    lw=0.25 + 1.7 * abs(scaled),
                    alpha=0.16 + 0.55 * abs(scaled),
                    transform=axis.transAxes,
                )

    active_layer = int(np.floor(progress * len(layer_sizes))) % len(layer_sizes)
    for layer_index, nodes in enumerate(positions):
        node_color = "#2563eb" if layer_index == 0 else "#8b5cf6" if layer_index == len(layer_sizes) - 1 else "#f59e0b"
        if layer_index == active_layer:
            node_color = "#0ea5e9"
        for x_pos, y_pos in nodes:
            axis.add_patch(
                Circle(
                    (x_pos, y_pos),
                    0.020,
                    fc=node_color,
                    ec="#0f172a",
                    lw=0.65,
                    alpha=0.96,
                    transform=axis.transAxes,
                )
            )

        layer_name = "Input" if layer_index == 0 else "Output" if layer_index == len(layer_sizes) - 1 else f"H{layer_index}"
        axis.text(
            nodes[0][0],
            0.92,
            layer_name,
            ha="center",
            va="center",
            fontsize=8,
            color="#334155",
            transform=axis.transAxes,
        )

    axis.text(0.50, 0.05, "Edge thickness = |weight|", ha="center", fontsize=8.2, color="#64748b", transform=axis.transAxes)


def _draw_nn_playground_classification_output(axis: plt.Axes, progress: float) -> None:
    rng = np.random.default_rng(123)
    class_a = rng.normal(loc=(-0.85, -0.55), scale=0.42, size=(120, 2))
    class_b = rng.normal(loc=(0.82, 0.62), scale=0.42, size=(120, 2))

    grid = np.linspace(-2.2, 2.2, 140)
    xx, yy = np.meshgrid(grid, grid)
    separator = xx * 1.0 - yy * 0.75 + 0.55 * np.sin(1.4 * xx)
    sharpness = 0.7 + 3.3 * progress
    probs = 1.0 / (1.0 + np.exp(-sharpness * separator))

    axis.contourf(xx, yy, probs, levels=np.linspace(0, 1, 9), cmap="RdYlBu", alpha=0.76)
    axis.contour(xx, yy, probs, levels=[0.5], colors="#0f172a", linewidths=1.3)
    axis.scatter(class_a[:, 0], class_a[:, 1], s=12, color="#2563eb", alpha=0.72, label="Class A")
    axis.scatter(class_b[:, 0], class_b[:, 1], s=12, color="#16a34a", alpha=0.72, label="Class B")
    axis.set_xlim(-2.2, 2.2)
    axis.set_ylim(-2.0, 2.0)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title("OUTPUT", fontsize=10, color="#0f172a", weight="bold")
    axis.text(0.5, -0.09, "Decision boundary sharpens", ha="center", fontsize=8, color="#334155", transform=axis.transAxes)
    axis.legend(loc="upper left", fontsize=7, frameon=False)


def _draw_nn_playground_regression_output(axis: plt.Axes, progress: float) -> None:
    rng = np.random.default_rng(246)
    x_train = np.linspace(-2.7, 2.7, 150)
    target = np.sin(1.25 * x_train) + 0.34 * x_train
    y_train = target + 0.17 * rng.normal(size=x_train.size)

    baseline = 0.22 * x_train
    prediction = (1 - progress) * baseline + progress * target

    axis.scatter(x_train, y_train, s=9, color="#94a3b8", alpha=0.58, label="train data")
    axis.plot(x_train, target, color="#2563eb", lw=1.6, alpha=0.65, label="target")
    axis.plot(x_train, prediction, color="#8b5cf6", lw=2.2, label="prediction")
    axis.set_xlim(-2.8, 2.8)
    axis.set_ylim(-2.0, 2.0)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title("OUTPUT", fontsize=10, color="#0f172a", weight="bold")
    axis.text(0.5, -0.09, "Function fit improves", ha="center", fontsize=8, color="#334155", transform=axis.transAxes)
    axis.legend(loc="upper left", fontsize=7, frameon=False)


@gif.frame
def _nn_playground_classification_frame(step: int, total_steps: int, dpi: int = 165) -> None:
    plt.close("all")
    fig = plt.figure(figsize=(10.6, 5.5), dpi=dpi)
    grid = fig.add_gridspec(1, 3, width_ratios=[0.27, 0.40, 0.33], wspace=0.08)
    controls_axis = fig.add_subplot(grid[0, 0])
    network_axis = fig.add_subplot(grid[0, 1])
    output_axis = fig.add_subplot(grid[0, 2])

    progress = step / max(1, total_steps - 1)
    epoch = int(np.interp(progress, [0, 1], [0, 1674]))
    train_loss = float(np.interp(progress, [0, 1], [0.502, 0.000]))
    test_loss = float(np.interp(progress, [0, 1], [0.511, 0.018]))

    _draw_nn_playground_controls(
        controls_axis,
        problem_type="Classification",
        epoch=epoch,
        train_loss=train_loss,
        test_loss=test_loss,
    )
    _draw_nn_playground_network(network_axis, progress=progress, seed=1)
    _draw_nn_playground_classification_output(output_axis, progress=progress)

    network_axis.set_title("3 HIDDEN LAYERS (6 → 4 → 2)", fontsize=10, color="#0f172a", weight="bold", pad=8)
    fig.suptitle("NN Playground Style — Classification", y=0.985, fontsize=14, color="#0f172a", weight="bold")
    fig.text(0.015, 0.015, "Edge Impulse • concept animation", fontsize=8, color="#64748b")


def render_nn_playground_classification(
    output_path: Path,
    frame_count: int = 30,
    hold_last: int = 8,
    duration_ms: int = 105,
    optimize: bool = True,
) -> Path:
    frames = [_nn_playground_classification_frame(step, frame_count) for step in range(frame_count)]
    frames.extend(
        [_nn_playground_classification_frame(frame_count - 1, frame_count) for _ in range(max(0, hold_last))]
    )
    gif.save(frames, str(output_path), duration=_slow_duration_ms(duration_ms))
    maybe_optimize_gif(output_path, optimize=optimize)
    return output_path


@gif.frame
def _nn_playground_regression_frame(step: int, total_steps: int, dpi: int = 165) -> None:
    plt.close("all")
    fig = plt.figure(figsize=(10.6, 5.5), dpi=dpi)
    grid = fig.add_gridspec(1, 3, width_ratios=[0.27, 0.40, 0.33], wspace=0.08)
    controls_axis = fig.add_subplot(grid[0, 0])
    network_axis = fig.add_subplot(grid[0, 1])
    output_axis = fig.add_subplot(grid[0, 2])

    progress = step / max(1, total_steps - 1)
    epoch = int(np.interp(progress, [0, 1], [0, 1800]))
    train_loss = float(np.interp(progress, [0, 1], [0.132, 0.122]))
    test_loss = float(np.interp(progress, [0, 1], [0.131, 0.122]))

    _draw_nn_playground_controls(
        controls_axis,
        problem_type="Regression",
        epoch=epoch,
        train_loss=train_loss,
        test_loss=test_loss,
    )
    _draw_nn_playground_network(network_axis, progress=progress, seed=2)
    _draw_nn_playground_regression_output(output_axis, progress=progress)

    network_axis.set_title("3 HIDDEN LAYERS (6 → 4 → 2)", fontsize=10, color="#0f172a", weight="bold", pad=8)
    fig.suptitle("NN Playground Style — Regression", y=0.985, fontsize=14, color="#0f172a", weight="bold")
    fig.text(0.015, 0.015, "Edge Impulse • concept animation", fontsize=8, color="#64748b")


def render_nn_playground_regression(
    output_path: Path,
    frame_count: int = 30,
    hold_last: int = 8,
    duration_ms: int = 105,
    optimize: bool = True,
) -> Path:
    frames = [_nn_playground_regression_frame(step, frame_count) for step in range(frame_count)]
    frames.extend(
        [_nn_playground_regression_frame(frame_count - 1, frame_count) for _ in range(max(0, hold_last))]
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

    fig.suptitle(title, x=0.5, y=0.985, ha="center", fontsize=16, fontweight="bold", color="#0f172a")

    start_y = 0.92
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
    fig.subplots_adjust(left=0.03, right=0.99, top=0.90, bottom=0.07, wspace=0.06)


def _style_concept_axis(axis: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    axis.set_title(title, fontsize=12, pad=8, color="#0f172a")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)


def classification_decision_boundary(x_values: np.ndarray) -> np.ndarray:
    return CLASSIFICATION_BOUNDARY_SLOPE * x_values + CLASSIFICATION_BOUNDARY_INTERCEPT


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
        boundary = classification_decision_boundary(x_values)
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
        height, width = 150, 220
        xx, yy = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
        concrete = 0.70 + 0.05 * np.sin(6.0 * xx + 4.5 * yy) + 0.06 * rng.normal(size=(height, width))
        concrete = np.clip(concrete, 0.45, 0.92)
        axis.imshow(concrete, cmap="gray", origin="lower", vmin=0, vmax=1)

        crack_x = np.array([32, 50, 69, 92, 114, 139, 164, 188], dtype=float)
        crack_y_base = np.array([126, 112, 101, 87, 73, 58, 44, 30], dtype=float)
        crack_y = crack_y_base + 1.8 * np.sin(np.linspace(0, np.pi, crack_x.size) + frame_seed * 0.35)
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
        axis.axis("off")
        axis.set_title("Transfer learning: image model before vs after", fontsize=12, color="#0f172a")

        left_axis = axis.inset_axes([0.05, 0.14, 0.40, 0.72])
        right_axis = axis.inset_axes([0.55, 0.14, 0.40, 0.72])

        base_labels = ["cat", "dog", "car", "other"]
        base_probs = np.array([0.27, 0.24, 0.23, 0.26]) + 0.012 * rng.normal(size=4)
        base_probs = np.clip(base_probs, 0.05, None)
        base_probs = base_probs / base_probs.sum()
        base_pos = np.arange(len(base_labels))
        left_axis.barh(base_pos, base_probs, color=["#cbd5e1"] * 4)
        left_axis.set_xlim(0, 1)
        left_axis.set_yticks(base_pos)
        left_axis.set_yticklabels(base_labels)
        left_axis.grid(alpha=0.2, axis="x")
        left_axis.set_title("Before transfer\n(generic base model)", fontsize=8, color="#334155")
        left_axis.tick_params(labelsize=7)
        left_axis.set_xlabel("Prob", fontsize=7)

        target_labels = ["gear", "bolt", "bearing", "other"]
        target_peak = float(np.clip(0.74 + 0.05 * np.sin(frame_seed * 0.35), 0.62, 0.85))
        remainder = 1.0 - target_peak
        target_probs = np.array([0.10, 0.12, target_peak, remainder - 0.22])
        target_probs = np.clip(target_probs, 0.03, None)
        target_probs = target_probs / target_probs.sum()
        target_pos = np.arange(len(target_labels))
        right_axis.barh(target_pos, target_probs, color=["#94a3b8", "#94a3b8", "#22c55e", "#94a3b8"])
        right_axis.set_xlim(0, 1)
        right_axis.set_yticks(target_pos)
        right_axis.set_yticklabels(target_labels)
        right_axis.grid(alpha=0.2, axis="x")
        right_axis.set_title("After transfer\n(fine-tuned classes)", fontsize=8, color="#166534")
        right_axis.tick_params(labelsize=7)
        right_axis.set_xlabel("Prob", fontsize=7)

        axis.annotate(
            "",
            xy=(0.55, 0.50),
            xytext=(0.45, 0.50),
            xycoords=axis.transAxes,
            arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#0ea5e9"},
        )
        axis.text(0.50, 0.54, "transfer +\nfine-tune", ha="center", va="bottom", fontsize=8, color="#0f766e", transform=axis.transAxes)
        return

    if block_name == "Keyword Spotting (Transfer Learning)":
        axis.axis("off")
        axis.set_title("Transfer learning: keyword spotting before vs after", fontsize=12, color="#0f172a")

        left_wave = axis.inset_axes([0.05, 0.58, 0.40, 0.28])
        right_wave = axis.inset_axes([0.55, 0.58, 0.40, 0.28])
        left_probs = axis.inset_axes([0.05, 0.14, 0.40, 0.30])
        right_probs = axis.inset_axes([0.55, 0.14, 0.40, 0.30])

        time = np.linspace(0, 1, 320)
        waveform = 0.45 * np.sin(2 * np.pi * 6 * time) + 0.22 * np.sin(2 * np.pi * 18 * time)
        waveform += 0.05 * rng.normal(size=time.size)

        left_wave.plot(time, waveform, color="#64748b", lw=1.1)
        left_wave.set_ylim(-1.2, 1.2)
        left_wave.set_xticks([])
        left_wave.set_yticks([])
        left_wave.set_title("Before transfer\n(generic speech model)", fontsize=8, color="#334155")
        left_wave.text(0.03, 0.08, "keyword score: 0.41", transform=left_wave.transAxes, fontsize=7.5, color="#475569")

        right_wave.plot(time, waveform, color="#0ea5e9", lw=1.1)
        right_wave.axvspan(0.56, 0.76, color="#22c55e", alpha=0.24)
        right_wave.set_ylim(-1.2, 1.2)
        right_wave.set_xticks([])
        right_wave.set_yticks([])
        keyword_score = float(np.clip(0.90 + 0.03 * np.sin(frame_seed * 0.25), 0.82, 0.96))
        right_wave.set_title("After transfer\n(wake-word fine-tune)", fontsize=8, color="#166534")
        right_wave.text(
            0.03,
            0.08,
            f"keyword score: {keyword_score:.2f}",
            transform=right_wave.transAxes,
            fontsize=7.5,
            color="#166534",
        )

        left_labels = ["speech", "music", "noise"]
        left_values = [0.37, 0.32, 0.31]
        left_pos = np.arange(len(left_labels))
        left_probs.barh(left_pos, left_values, color=["#cbd5e1", "#cbd5e1", "#cbd5e1"])
        left_probs.set_xlim(0, 1)
        left_probs.set_yticks(left_pos)
        left_probs.set_yticklabels(left_labels)
        left_probs.grid(alpha=0.2, axis="x")
        left_probs.tick_params(labelsize=7)
        left_probs.set_xlabel("Prob", fontsize=7)

        right_labels = ["hey-edge", "other", "silence"]
        right_values = [keyword_score, 0.07, max(0.01, 1.0 - keyword_score - 0.07)]
        right_pos = np.arange(len(right_labels))
        right_probs.barh(right_pos, right_values, color=["#22c55e", "#94a3b8", "#94a3b8"])
        right_probs.set_xlim(0, 1)
        right_probs.set_yticks(right_pos)
        right_probs.set_yticklabels(right_labels)
        right_probs.grid(alpha=0.2, axis="x")
        right_probs.tick_params(labelsize=7)
        right_probs.set_xlabel("Prob", fontsize=7)

        axis.annotate(
            "",
            xy=(0.55, 0.50),
            xytext=(0.45, 0.50),
            xycoords=axis.transAxes,
            arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#0ea5e9"},
        )
        axis.text(0.50, 0.54, "transfer +\nfine-tune", ha="center", va="bottom", fontsize=8, color="#0f766e", transform=axis.transAxes)
        return

    if block_name == "Object Detection (MobileNetV2 SSD FPN)":
        canvas = np.tile(np.linspace(0.90, 0.76, 220), (150, 1))
        axis.imshow(canvas, cmap="gray", origin="lower", vmin=0, vmax=1)

        axis.add_patch(Rectangle((0, 0), 220, 38, fc="#d6d3d1", ec="none", alpha=0.95))

        apple_center = np.array([62.0, 64.0]) + rng.normal(scale=[0.7, 0.5], size=2)
        axis.add_patch(Circle((apple_center[0], apple_center[1]), radius=16, fc="#ef4444", ec="#991b1b", lw=1.4, alpha=0.95))
        axis.add_patch(Rectangle((apple_center[0] - 1.4, apple_center[1] + 14), 2.8, 8, fc="#7c2d12", ec="none"))
        axis.add_patch(Ellipse((apple_center[0] + 8, apple_center[1] + 18), width=10, height=5, angle=35, fc="#22c55e", ec="#166534", lw=0.8))

        orange_center = np.array([148.0, 58.0]) + rng.normal(scale=[0.8, 0.6], size=2)
        axis.add_patch(Circle((orange_center[0], orange_center[1]), radius=15, fc="#f97316", ec="#9a3412", lw=1.4, alpha=0.95))
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
        footer = "Edge Impulse processing blocks"
        _draw_dsp_block_concept(concept_axis, block_name, frame_seed=step)
    else:
        title = f"ML Learning Block: {block_name}"
        footer = "Edge Impulse learning blocks"
        _draw_ml_block_concept(concept_axis, block_name, frame_seed=step)

    fig.suptitle(title, x=0.5, y=0.985, ha="center", fontsize=14, fontweight="bold", color="#0f172a")
    fig.text(0.02, 0.02, footer, ha="left", fontsize=8.8, color="#64748b")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.13)


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
        elif preset == "nn_training_layers":
            generated.append(
                render_nn_training_layers(
                    output_path=output_path,
                    frame_count=24,
                    hold_last=6,
                    duration_ms=95,
                    optimize=optimize,
                )
            )
        elif preset == "nn_inference_layers":
            generated.append(
                render_nn_inference_layers(
                    output_path=output_path,
                    frame_count=24,
                    hold_last=6,
                    duration_ms=95,
                    optimize=optimize,
                )
            )
        elif preset == "nn_training_vs_on_device_inference":
            generated.append(
                render_nn_training_vs_on_device_inference(
                    output_path=output_path,
                    frame_count=24,
                    hold_last=6,
                    duration_ms=95,
                    optimize=optimize,
                )
            )
        elif preset == "nn_single_neuron":
            generated.append(
                render_nn_single_neuron(
                    output_path=output_path,
                    frame_count=28,
                    hold_last=8,
                    duration_ms=100,
                    optimize=optimize,
                )
            )
        elif preset == "nn_architecture_layers":
            generated.append(
                render_nn_architecture_layers(
                    output_path=output_path,
                    frame_count=26,
                    hold_last=8,
                    duration_ms=100,
                    optimize=optimize,
                )
            )
        elif preset == "nn_deep_network":
            generated.append(
                render_nn_deep_network(
                    output_path=output_path,
                    frame_count=30,
                    hold_last=8,
                    duration_ms=95,
                    optimize=optimize,
                )
            )
        elif preset == "nn_backpropagation_learning":
            generated.append(
                render_nn_backpropagation_learning(
                    output_path=output_path,
                    frame_count=48,
                    hold_last=12,
                    duration_ms=120,
                    optimize=optimize,
                )
            )
        elif preset == "nn_playground_classification":
            generated.append(
                render_nn_playground_classification(
                    output_path=output_path,
                    frame_count=30,
                    hold_last=8,
                    duration_ms=105,
                    optimize=optimize,
                )
            )
        elif preset == "nn_playground_regression":
            generated.append(
                render_nn_playground_regression(
                    output_path=output_path,
                    frame_count=30,
                    hold_last=8,
                    duration_ms=105,
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
        elif preset == "embedded_quantization_8bit_vs_float32":
            generated.append(
                render_embedded_quantization_8bit_vs_float32(
                    output_path=output_path,
                    frame_count=22,
                    hold_last=6,
                    duration_ms=95,
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