from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

import animation_pipeline as pipeline


def test_ensure_output_dirs_creates_directories(tmp_path: Path) -> None:
    img_dir = tmp_path / "img"
    scratch_dir = tmp_path / "scratch"

    pipeline.ensure_output_dirs(img_dir=img_dir, scratch_dir=scratch_dir)

    assert img_dir.exists()
    assert scratch_dir.exists()


def test_render_sine_bead_creates_gif(tmp_path: Path) -> None:
    output_path = tmp_path / "sine_bead.gif"

    created = pipeline.render_sine_bead(
        output_path=output_path,
        frame_count=6,
        hold_last=1,
        duration_ms=40,
        optimize=False,
    )

    assert created == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_render_nn_sigmoid_shift_creates_gif(tmp_path: Path) -> None:
    output_path = tmp_path / "nn_sigmoid_shift.gif"

    created = pipeline.render_nn_sigmoid_shift(
        output_path=output_path,
        frame_count=6,
        hold_last=1,
        duration_ms=40,
        optimize=False,
    )

    assert created == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_render_dsp_sine_shift_creates_gif(tmp_path: Path) -> None:
    output_path = tmp_path / "dsp_sine_shift.gif"
    scratch_dir = tmp_path / "scratch"

    created = pipeline.render_dsp_sine_shift(
        output_path=output_path,
        scratch_dir=scratch_dir,
        frame_count=6,
        fps=4,
        optimize=False,
    )

    assert created == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_build_colour_mapping_returns_continent_keys() -> None:
    data_frame = pd.DataFrame(
        {
            "Continent": ["Asia", "Europe", "Asia", "Africa"],
            "Year": [2017, 2017, 2018, 2018],
            "GDP per capita": [1200, 5500, 1300, 1400],
            "Life expectancy": [70, 78, 71, 68],
            "Population": [10_000, 15_000, 10_200, 9_000],
        }
    )

    mapping = pipeline.build_colour_mapping(data_frame)

    assert set(mapping.keys()) == {"Africa", "Asia", "Europe"}
    assert all(isinstance(value, str) for value in mapping.values())


def test_render_gapminder_full_creates_gif_from_dataframe(tmp_path: Path) -> None:
    output_path = tmp_path / "gapminder_full.gif"
    data_frame = pd.DataFrame(
        {
            "Country": ["A", "B", "A", "B"],
            "Continent": ["Asia", "Europe", "Asia", "Europe"],
            "Year": [2017, 2017, 2018, 2018],
            "GDP per capita": [1200, 5500, 1300, 5700],
            "Life expectancy": [70, 78, 71, 79],
            "Population": [10_000, 15_000, 10_200, 15_300],
        }
    )

    created = pipeline.render_gapminder_full(
        output_path=output_path,
        data_frame=data_frame,
        start_year=2017,
        end_year=2018,
        hold_last=1,
        duration_ms=40,
        optimize=False,
    )

    assert created == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_run_selected_presets_with_single_preset(tmp_path: Path) -> None:
    img_dir = tmp_path / "img"
    scratch_dir = tmp_path / "scratch"

    outputs = pipeline.run_selected_presets(
        presets=["sine_bead"],
        img_dir=img_dir,
        scratch_dir=scratch_dir,
        optimize=False,
        fps=4,
    )

    assert len(outputs) == 1
    assert outputs[0].exists()