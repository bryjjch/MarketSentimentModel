"""
Build ``model.tar.gz`` for SageMaker Hugging Face hosting.

Copies a Hugging Face save directory (config, weights, tokenizer files) plus ``code/inference.py``
into a tarball layout expected by the HF inference DLC.

Example (from repo root, after training):

  python infra/scripts/package_model_tarball.py --model_dir outputs/clf_finbert --output model.tar.gz
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_serving_code_dir() -> Path:
    return _repo_root() / "infra" / "sagemaker" / "serving" / "code"


def _ignore_checkpoints(directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if name.startswith("checkpoint-"):
            ignored.add(name)
    return ignored


def package_model(
    model_dir: Path,
    output_path: Path,
    *,
    serving_code_dir: Path,
) -> None:
    model_dir = model_dir.resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"model_dir is not a directory: {model_dir}")

    serving_code_dir = serving_code_dir.resolve()
    inference_py = serving_code_dir / "inference.py"
    if not inference_py.is_file():
        raise SystemExit(f"Missing serving script: {inference_py}")

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="finsense-model-tar-") as tmp:
        stage = Path(tmp) / "bundle"
        stage.mkdir(parents=True)
        # HF weights + tokenizer at tarball root
        shutil.copytree(
            model_dir,
            stage,
            dirs_exist_ok=True,
            ignore=_ignore_checkpoints,
        )
        code_dst = stage / "code"
        if code_dst.exists():
            shutil.rmtree(code_dst)
        shutil.copytree(serving_code_dir, code_dst)

        cwd = Path.cwd()
        try:
            os.chdir(stage)
            with tarfile.open(output_path, "w:gz") as tf:
                for name in sorted(os.listdir(".")):
                    tf.add(name, arcname=name.replace("\\", "/"), recursive=True)
        finally:
            os.chdir(cwd)

    print(f"Wrote {output_path} ({output_path.stat().st_size // 1024} KiB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Hugging Face model directory (e.g. outputs/clf_finbert)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model.tar.gz"),
        help="Output tarball path (default: ./model.tar.gz)",
    )
    parser.add_argument(
        "--serving_code",
        type=Path,
        default=_default_serving_code_dir(),
        help="Directory containing inference.py (default: infra/sagemaker/serving/code)",
    )
    args = parser.parse_args()
    try:
        package_model(args.model_dir, args.output, serving_code_dir=args.serving_code)
    except SystemExit as e:
        print(e, file=sys.stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
