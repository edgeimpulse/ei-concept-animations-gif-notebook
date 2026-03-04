# Edge Impulse Concept Animations — GIF Notebook

A Jupyter notebook for generating animated GIFs that illustrate key Edge Impulse and machine learning pipeline concepts. These animations are intended for use in documentation, tutorials, and educational materials.

## Overview

This project provides a self-contained notebook that:

- Renders step-by-step animated visualisations of common ML/DSP concepts (e.g. signal sampling, windowing, feature extraction, model inference)
- Exports each animation as a looping GIF ready for embedding in web pages, markdown docs, or slide decks
- Keeps every animation reproducible and parameterised so colours, timing, and content can be adjusted without rewriting the underlying drawing code

## Prerequisites

| Requirement | Recommended version |
|---|---|
| Python | 3.9 + |
| Jupyter Notebook / JupyterLab | latest |
| `matplotlib` | 3.7 + |
| `Pillow` | 10 + |
| `numpy` | 1.24 + |

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/edgeimpulse/ei-concept-animations-gif-notebook.git
   cd ei-concept-animations-gif-notebook
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Launch Jupyter and open the notebook:

```bash
jupyter notebook concept_animations.ipynb
```

Run all cells from top to bottom (`Kernel → Restart & Run All`). Each animation section:

1. Defines the frames using matplotlib figures
2. Saves the frames to a temporary directory
3. Stitches them into a GIF using Pillow and writes the file to `output/`

### Generating a single animation

Every animation is wrapped in a helper function so it can be called independently:

```python
from animations.sampling import render_sampling_animation

render_sampling_animation(output_path="output/sampling.gif", fps=10)
```

### Customising an animation

Key parameters (frame count, colours, labels, figure size) are exposed as function arguments or top-of-cell constants. Adjust them before re-running the relevant cell.

## Output

All GIFs are written to the `output/` directory. File names match the concept they illustrate, e.g.:

```
output/
├── sampling.gif
├── windowing.gif
├── dsp_pipeline.gif
├── neural_network_inference.gif
└── ...
```

## Project Structure

```
ei-concept-animations-gif-notebook/
├── concept_animations.ipynb   # Main notebook
├── animations/                # Animation helper modules
│   └── ...
├── output/                    # Generated GIFs (git-ignored)
├── requirements.txt
└── README.md
```

## Contributing

1. Fork the repository and create a feature branch.
2. Add or update an animation, keeping the helper-function pattern consistent with existing animations.
3. Confirm that all GIFs render correctly by running the full notebook.
4. Open a pull request with a brief description of what changed and why.

## License

See [LICENSE](LICENSE) for details.
