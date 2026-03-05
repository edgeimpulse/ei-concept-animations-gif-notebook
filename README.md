# Edge Impulse Concept Animations — ML + DSP GIF Toolkit

This repository provides a practical workflow for creating animated visualizations of machine learning and digital signal processing concepts.

It includes:

- Notebook explorations for rapid prototyping
- Python scripts for repeatable batch generation
- Ready-to-copy templates for creating your own concept animations
- A documented structure so contributors can quickly add new ideas

## Quick Start

### 1) Create environment

```bash
/opt/homebrew/bin/python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r python/requirements.txt
```

### 2) Install gifsicle (optional, for optimization)

```bash
brew install gifsicle
```

### 3) Generate animations

```bash
python python/animation_pipeline.py --all
```

Generated files are written to `img/` and temporary frame PNGs to `scratch/`.

## Animation Presets

The script `python/animation_pipeline.py` includes these presets:

- `dsp_sine_shift` — moving sine-wave phase animation (DSP)
- `dsp_processing_blocks` — Edge Impulse processing blocks walkthrough (DSP)
- `dsp_processing_blocks_individual` — one standalone GIF per DSP processing block concept
- `nn_sigmoid_shift` — shifting activation curve animation (ML)
- `nn_training_layers` — NN training walkthrough showing layer creation and weight updates
- `nn_inference_layers` — NN inference walkthrough showing forward pass through fixed layers
- `nn_training_vs_on_device_inference` — side-by-side cloud training vs on-device inference
- `nn_single_neuron` — single neuron animation showing inputs, weights, bias, and activation
- `nn_architecture_layers` — neural network architecture showing input, hidden, and output layers
- `nn_deep_network` — deep neural network with multiple hidden layers
- `nn_backpropagation_learning` — backpropagation learning process with forward and backward passes
- `nn_playground_classification` — TensorFlow Playground-style classification animation (nn-1/nn-2 inspired)
- `nn_playground_regression` — TensorFlow Playground-style regression animation (nn-3/nn-4 inspired)
- `ml_learning_blocks` — Edge Impulse learning blocks walkthrough (ML)
- `ml_learning_blocks_individual` — one standalone GIF per ML learning block concept
- `embedded_quantization_8bit_vs_float32` — embedded quantization comparison (8-bit vs float32)
- `sine_bead` — bead moving over a sine curve (concept intro)
- `gapminder_full` — complete Gapminder sequence using `gif`

Run one preset:

```bash
python python/animation_pipeline.py --preset dsp_processing_blocks
```

Run all presets:

```bash
python python/animation_pipeline.py --all
```

Generate one animation per DSP processing block:

```bash
python python/animation_pipeline.py --preset dsp_processing_blocks_individual
```

Generate one animation per ML learning block:

```bash
python python/animation_pipeline.py --preset ml_learning_blocks_individual
```

Generate NN training/inference samples:

```bash
python python/animation_pipeline.py --preset nn_training_layers
python python/animation_pipeline.py --preset nn_inference_layers
python python/animation_pipeline.py --preset nn_training_vs_on_device_inference
```

Generate NN concept animations:

```bash
python python/animation_pipeline.py --preset nn_single_neuron
python python/animation_pipeline.py --preset nn_architecture_layers
python python/animation_pipeline.py --preset nn_deep_network
python python/animation_pipeline.py --preset nn_backpropagation_learning
python python/animation_pipeline.py --preset nn_playground_classification
python python/animation_pipeline.py --preset nn_playground_regression
```

Generate embedded quantization sample:

```bash
python python/animation_pipeline.py --preset embedded_quantization_8bit_vs_float32
```

## Reference Samples by Type

### DSP

| Sample | Preview |
|---|---|
| `dsp_sine_shift` | ![dsp_sine_shift](./img/dsp_sine_shift.gif) |

### ML

| Sample | Preview |
|---|---|
| `nn_sigmoid_shift` | ![nn_sigmoid_shift](./img/nn_sigmoid_shift.gif) |
| `nn_training_layers` | ![nn_training_layers](./img/nn_training_layers.gif) |
| `nn_inference_layers` | ![nn_inference_layers](./img/nn_inference_layers.gif) |
| `nn_training_vs_on_device_inference` | ![nn_training_vs_on_device_inference](./img/nn_training_vs_on_device_inference.gif) |
| `nn_single_neuron` | ![nn_single_neuron](./img/nn_single_neuron.gif) |
| `nn_architecture_layers` | ![nn_architecture_layers](./img/nn_architecture_layers.gif) |
| `nn_deep_network` | ![nn_deep_network](./img/nn_deep_network.gif) |
| `nn_backpropagation_learning` | ![nn_backpropagation_learning](./img/nn_backpropagation_learning.gif) |
| `nn_playground_classification` | ![nn_playground_classification](./img/nn_playground_classification.gif) |
| `nn_playground_regression` | ![nn_playground_regression](./img/nn_playground_regression.gif) |
| `sine_bead` | ![sine_bead](./img/sine_bead.gif) |
| `gapminder_full` | ![gapminder_full](./img/gapminder_full.gif) |

### Embedded

| Sample | Preview |
|---|---|
| `embedded_quantization_8bit_vs_float32` | ![embedded_quantization_8bit_vs_float32](./img/embedded_quantization_8bit_vs_float32.gif) |

## Individual Block Previews

### DSP Processing Blocks

| Block | Preview | Concept Accuracy (%) |
|---|---|---|
| Raw Data | ![dsp_block_raw_data](./img/dsp_block_raw_data.gif) | 23.6% |
| Flatten | ![dsp_block_flatten](./img/dsp_block_flatten.gif) | 25.1% |
| Image | ![dsp_block_image](./img/dsp_block_image.gif) | 86.8% |
| Spectral features | ![dsp_block_spectral_features](./img/dsp_block_spectral_features.gif) | 46.6% |
| Spectrogram | ![dsp_block_spectrogram](./img/dsp_block_spectrogram.gif) | 84.7% |
| Audio MFE | ![dsp_block_audio_mfe](./img/dsp_block_audio_mfe.gif) | 73.7% |
| Audio MFCC | ![dsp_block_audio_mfcc](./img/dsp_block_audio_mfcc.gif) | 36.0% |
| Audio Syntiant | ![dsp_block_audio_syntiant](./img/dsp_block_audio_syntiant.gif) | 58.6% |
| IMU Syntiant | ![dsp_block_imu_syntiant](./img/dsp_block_imu_syntiant.gif) | 24.2% |
| HR/HRV features | ![dsp_block_hr_hrv_features](./img/dsp_block_hr_hrv_features.gif) | 20.9% |

### ML Learning Blocks

| Block | Preview | Concept Accuracy (%) |
|---|---|---|
| Classification (Keras) | ![ml_block_classification_keras](./img/ml_block_classification_keras.gif) | 38.1% |
| Regression (Keras) | ![ml_block_regression_keras](./img/ml_block_regression_keras.gif) | 23.9% |
| Anomaly Detection (K-means) | ![ml_block_anomaly_detection_k_means](./img/ml_block_anomaly_detection_k_means.gif) | 36.4% |
| Anomaly Detection (GMM) | ![ml_block_anomaly_detection_gmm](./img/ml_block_anomaly_detection_gmm.gif) | 39.7% |
| Visual anomaly detection (FOMO-AD) | ![ml_block_visual_anomaly_detection_fomo_ad](./img/ml_block_visual_anomaly_detection_fomo_ad.gif) | 87.9% |
| Image Classification (Transfer Learning) | ![ml_block_image_classification_transfer_learning](./img/ml_block_image_classification_transfer_learning.gif) | 61.0% |
| Keyword Spotting (Transfer Learning) | ![ml_block_keyword_spotting_transfer_learning](./img/ml_block_keyword_spotting_transfer_learning.gif) | 26.0% |
| Object Detection (MobileNetV2 SSD FPN) | ![ml_block_object_detection_mobilenetv2_ssd_fpn](./img/ml_block_object_detection_mobilenetv2_ssd_fpn.gif) | 51.2% |
| Object Detection (FOMO) | ![ml_block_object_detection_fomo](./img/ml_block_object_detection_fomo.gif) | 82.1% |
| Classical ML | ![ml_block_classical_ml](./img/ml_block_classical_ml.gif) | 16.3% |
| Custom block (PyTorch/Keras/scikit-learn) | ![ml_block_custom_block_pytorch_keras_scikit_learn](./img/ml_block_custom_block_pytorch_keras_scikit_learn.gif) | 47.4% |

## Running the Notebooks

Each notebook can be run independently by opening it in Jupyter Notebook or JupyterLab and executing cells sequentially.

Main notebook:

- `notebook/animations_vis_animation.ipynb`

## Output

All generated animations are saved to the `img/` directory. Temporary PNG frames are written to `scratch/` for image-sequence workflows.

## Build Your Own Animation

Start from one of the templates:

- `python/templates/dsp_template.py`
- `python/templates/ml_template.py`

Then follow the contributor workflow in `docs/BUILD_YOUR_OWN.md`.

## GIF Style Guide

All GIF concepts should follow these shared visual rules:

- Centered header in the top band of each frame (clear concept title)
- Watermark anchored to the bottom-left corner
- Consistent palette using the core tokens: `#0f172a`, `#334155`, `#64748b`, `#2563eb`, `#10b981`, `#8b5cf6`, `#f59e0b`, `#dc2626`
- Red (`#dc2626`) reserved for point-of-interest cues (anomalies, detections, important updates)

Run style guide checking across all GIFs:

```bash
python python/concept_validation.py --img-dir img --style-check --style-json-out scratch/style_validation_results.json
```

The style report includes per-GIF checks for header placement, watermark placement, palette alignment, and red point-of-interest usage.

## Testing

Run automated tests:

```bash
python -m pytest -q
```

Run concept validation (generates one reference script per concept, scores GIF-to-concept accuracy, and updates README accuracy columns):

```bash
python python/concept_validation.py --img-dir img --generate-scripts --update-readme --json-out scratch/concept_validation_results.json
```

Generated reference scripts are written to `scratch/concept_validation_scripts/`.

## Legacy Notebook Exports

The files below are preserved for compatibility/reference:

- `python/animations_vis_animation.py`
- `python/dsp_and_nn_animations_vis_animation (2).py`

They are notebook exports and not the recommended starting point for new contributions.
