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
- `ml_learning_blocks` — Edge Impulse learning blocks walkthrough (ML)
- `ml_learning_blocks_individual` — one standalone GIF per ML learning block concept
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

## GIF Previews

| Preset | Preview |
|---|---|
| `dsp_sine_shift` | ![dsp_sine_shift](./img/dsp_sine_shift.gif) |
| `nn_sigmoid_shift` | ![nn_sigmoid_shift](./img/nn_sigmoid_shift.gif) |
| `sine_bead` | ![sine_bead](./img/sine_bead.gif) |
| `gapminder_full` | ![gapminder_full](./img/gapminder_full.gif) |

## Individual Block Previews

### DSP Processing Blocks

| Block | Preview |
|---|---|
| Raw Data | ![dsp_block_raw_data](./img/dsp_block_raw_data.gif) |
| Flatten | ![dsp_block_flatten](./img/dsp_block_flatten.gif) |
| Image | ![dsp_block_image](./img/dsp_block_image.gif) |
| Spectral features | ![dsp_block_spectral_features](./img/dsp_block_spectral_features.gif) |
| Spectrogram | ![dsp_block_spectrogram](./img/dsp_block_spectrogram.gif) |
| Audio MFE | ![dsp_block_audio_mfe](./img/dsp_block_audio_mfe.gif) |
| Audio MFCC | ![dsp_block_audio_mfcc](./img/dsp_block_audio_mfcc.gif) |
| Audio Syntiant | ![dsp_block_audio_syntiant](./img/dsp_block_audio_syntiant.gif) |
| IMU Syntiant | ![dsp_block_imu_syntiant](./img/dsp_block_imu_syntiant.gif) |
| HR/HRV features | ![dsp_block_hr_hrv_features](./img/dsp_block_hr_hrv_features.gif) |

### ML Learning Blocks

| Block | Preview |
|---|---|
| Classification (Keras) | ![ml_block_classification_keras](./img/ml_block_classification_keras.gif) |
| Regression (Keras) | ![ml_block_regression_keras](./img/ml_block_regression_keras.gif) |
| Anomaly Detection (K-means) | ![ml_block_anomaly_detection_k_means](./img/ml_block_anomaly_detection_k_means.gif) |
| Anomaly Detection (GMM) | ![ml_block_anomaly_detection_gmm](./img/ml_block_anomaly_detection_gmm.gif) |
| Visual anomaly detection (FOMO-AD) | ![ml_block_visual_anomaly_detection_fomo_ad](./img/ml_block_visual_anomaly_detection_fomo_ad.gif) |
| Image Classification (Transfer Learning) | ![ml_block_image_classification_transfer_learning](./img/ml_block_image_classification_transfer_learning.gif) |
| Keyword Spotting (Transfer Learning) | ![ml_block_keyword_spotting_transfer_learning](./img/ml_block_keyword_spotting_transfer_learning.gif) |
| Object Detection (MobileNetV2 SSD FPN) | ![ml_block_object_detection_mobilenetv2_ssd_fpn](./img/ml_block_object_detection_mobilenetv2_ssd_fpn.gif) |
| Object Detection (FOMO) | ![ml_block_object_detection_fomo](./img/ml_block_object_detection_fomo.gif) |
| Classical ML | ![ml_block_classical_ml](./img/ml_block_classical_ml.gif) |
| Custom block (PyTorch/Keras/scikit-learn) | ![ml_block_custom_block_pytorch_keras_scikit_learn](./img/ml_block_custom_block_pytorch_keras_scikit_learn.gif) |

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

## Testing

Run automated tests:

```bash
python -m pytest -q
```

## Legacy Notebook Exports

The files below are preserved for compatibility/reference:

- `python/animations_vis_animation.py`
- `python/dsp_and_nn_animations_vis_animation (2).py`

They are notebook exports and not the recommended starting point for new contributions.
