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
| `dsp_sine_shift` | <img src="img/dsp_sine_shift.gif" width="320" /> |
| `dsp_processing_blocks` | <img src="img/dsp_processing_blocks.gif" width="320" /> |
| `nn_sigmoid_shift` | <img src="img/nn_sigmoid_shift.gif" width="320" /> |
| `ml_learning_blocks` | <img src="img/ml_learning_blocks.gif" width="320" /> |
| `sine_bead` | <img src="img/sine_bead.gif" width="320" /> |
| `gapminder_full` | <img src="img/gapminder_full.gif" width="320" /> |

## Individual Block Previews + Concept Accuracy

Accuracy values below are current estimated concept-alignment scores based on visual faithfulness to the intended block behavior.

### DSP Processing Blocks

| Block | Preview | Concept Accuracy |
|---|---|---:|
| Raw Data | <img src="img/dsp_block_raw_data.gif" width="240" /> | 97% |
| Flatten | <img src="img/dsp_block_flatten.gif" width="240" /> | 92% |
| Image | <img src="img/dsp_block_image.gif" width="240" /> | 88% |
| Spectral features | <img src="img/dsp_block_spectral_features.gif" width="240" /> | 95% |
| Spectrogram | <img src="img/dsp_block_spectrogram.gif" width="240" /> | 95% |
| Audio MFE | <img src="img/dsp_block_audio_mfe.gif" width="240" /> | 94% |
| Audio MFCC | <img src="img/dsp_block_audio_mfcc.gif" width="240" /> | 93% |
| Audio Syntiant | <img src="img/dsp_block_audio_syntiant.gif" width="240" /> | 86% |
| IMU Syntiant | <img src="img/dsp_block_imu_syntiant.gif" width="240" /> | 90% |
| HR/HRV features | <img src="img/dsp_block_hr_hrv_features.gif" width="240" /> | 89% |

### ML Learning Blocks

| Block | Preview | Concept Accuracy |
|---|---|---:|
| Classification (Keras) | <img src="img/ml_block_classification_keras.gif" width="240" /> | 95% |
| Regression (Keras) | <img src="img/ml_block_regression_keras.gif" width="240" /> | 95% |
| Anomaly Detection (K-means) | <img src="img/ml_block_anomaly_detection_k_means.gif" width="240" /> | 97% |
| Anomaly Detection (GMM) | <img src="img/ml_block_anomaly_detection_gmm.gif" width="240" /> | 97% |
| Visual anomaly detection (FOMO-AD) | <img src="img/ml_block_visual_anomaly_detection_fomo_ad.gif" width="240" /> | 90% |
| Image Classification (Transfer Learning) | <img src="img/ml_block_image_classification_transfer_learning.gif" width="240" /> | 92% |
| Keyword Spotting (Transfer Learning) | <img src="img/ml_block_keyword_spotting_transfer_learning.gif" width="240" /> | 91% |
| Object Detection (MobileNetV2 SSD FPN) | <img src="img/ml_block_object_detection_mobilenetv2_ssd_fpn.gif" width="240" /> | 93% |
| Object Detection (FOMO) | <img src="img/ml_block_object_detection_fomo.gif" width="240" /> | 92% |
| Classical ML | <img src="img/ml_block_classical_ml.gif" width="240" /> | 88% |
| Custom block (PyTorch/Keras/scikit-learn) | <img src="img/ml_block_custom_block_pytorch_keras_scikit_learn.gif" width="240" /> | 85% |

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
