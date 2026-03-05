from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


pipeline = _load_module("animation_pipeline", PROJECT_ROOT / "python" / "animation_pipeline.py")
validation = _load_module("concept_validation", PROJECT_ROOT / "python" / "concept_validation.py")


def test_classification_boundary_separates_classes() -> None:
    rng = np.random.default_rng(500)
    class_a = rng.normal(loc=(-1.0, -0.3), scale=0.35, size=(200, 2))
    class_b = rng.normal(loc=(0.9, 0.8), scale=0.35, size=(200, 2))

    class_a_boundary = pipeline.classification_decision_boundary(class_a[:, 0])
    class_b_boundary = pipeline.classification_decision_boundary(class_b[:, 0])

    class_a_below_rate = np.mean(class_a[:, 1] < class_a_boundary)
    class_b_above_rate = np.mean(class_b[:, 1] > class_b_boundary)

    assert class_a_below_rate > 0.8
    assert class_b_above_rate > 0.8


def test_validate_all_concepts_and_generate_reference_scripts(tmp_path: Path) -> None:
    img_dir = tmp_path / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    pipeline.render_all_dsp_processing_block_animations(
        img_dir=img_dir,
        frame_count=3,
        hold_last=0,
        duration_ms=20,
        optimize=False,
    )
    pipeline.render_all_ml_learning_block_animations(
        img_dir=img_dir,
        frame_count=3,
        hold_last=0,
        duration_ms=20,
        optimize=False,
    )

    scripts_dir = tmp_path / "scripts"
    scripts = validation.generate_reference_scripts(scripts_dir)

    assert len(scripts) == len(validation.build_concept_specs())
    assert all(path.exists() for path in scripts)

    results = validation.validate_all_concepts(img_dir=img_dir)

    expected_count = len(pipeline.DSP_PROCESSING_BLOCKS) + len(pipeline.ML_LEARNING_BLOCKS)
    assert len(results) == expected_count

    scores = [result.accuracy_percent for result in results if result.accuracy_percent is not None]
    assert len(scores) == expected_count
    assert min(scores) >= 15.0

    classification = next(result for result in results if result.block_name == "Classification (Keras)")
    assert classification.accuracy_percent is not None
    assert classification.accuracy_percent >= 30.0


def test_validate_style_for_all_gifs_and_json_output(tmp_path: Path) -> None:
    img_dir = tmp_path / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    pipeline.render_nn_single_neuron(
        output_path=img_dir / "nn_single_neuron.gif",
        frame_count=4,
        hold_last=0,
        duration_ms=20,
        optimize=False,
    )
    pipeline.render_embedded_quantization_8bit_vs_float32(
        output_path=img_dir / "embedded_quantization_8bit_vs_float32.gif",
        frame_count=4,
        hold_last=0,
        duration_ms=20,
        optimize=False,
    )

    style_results = validation.validate_style_for_all_gifs(img_dir=img_dir)

    assert len(style_results) == 2
    assert all(0.0 <= result.style_score_percent <= 100.0 for result in style_results)

    payload = validation.style_results_to_json(style_results)
    assert "style_guide" in payload
    assert "summary" in payload
    assert payload["summary"]["total_gifs"] == 2
    assert len(payload["results"]) == 2
