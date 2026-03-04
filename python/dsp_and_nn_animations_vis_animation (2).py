"""Legacy compatibility entrypoint.

This wrapper preserves the historical filename while delegating generation
to the maintained pipeline in `animation_pipeline.py`.
"""

from animation_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main(["--all"]))