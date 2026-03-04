# Build Your Own Animation (ML + DSP)

This guide is the recommended contributor path for adding new concept animations.

## 1) Choose a template

- DSP-style animation: copy `python/templates/dsp_template.py`
- ML-style animation: copy `python/templates/ml_template.py`

Name the new file based on the concept, for example:

- `python/templates/dsp_fft_windowing.py`
- `python/templates/ml_softmax_temperature.py`

## 2) Implement your frame logic

For DSP concepts, update the `build_frame_figure` function to draw one frame for a given `step`.

For ML concepts, update the `build_frame` function (decorated with `@gif.frame`) to draw one frame for a given `step`.

Keep frame generation deterministic:

- avoid randomness unless seeded
- make axis limits stable across frames
- use readable titles and labels

## 3) Add a preset to the pipeline

Open `python/animation_pipeline.py` and add your preset in two places:

1. Add the preset name in `PRESET_NAMES`
2. Add a branch in `run_selected_presets()` that calls your renderer

Use clear IDs such as:

- `dsp_stft_bins`
- `ml_attention_weights`

## 4) Generate and review

Run one preset while iterating:

```bash
python python/animation_pipeline.py --preset <your_preset_name>
```

Run the full set before opening a PR:

```bash
python python/animation_pipeline.py --all
```

Outputs go to `img/` and temporary PNGs to `scratch/`.

## 5) Test and submit

Run tests:

```bash
python -m pytest -q
```

In your PR include:

- the new preset name
- a short description of the concept
- generated GIF path(s)

## Suggested quality checklist

- Animation loops smoothly
- Last frame is readable and intentional
- Visual encoding is consistent (colors/labels)
- No dependency changes unless required
- README/docs updated if behavior changed