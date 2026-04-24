"""Thin SageMaker entry-point wrapper for classifier fine-tuning.

The ``training`` package is copied into the source tarball via the estimator's
``dependencies`` parameter so that relative imports resolve normally.

When the pipeline feeds an MLM model artifact via a training channel, the data
arrives as a ``model.tar.gz``.  This wrapper extracts it into a flat directory
so ``--mlm_checkpoint`` resolves to real model files.
"""

import os
import tarfile
from pathlib import Path


def _extract_channel_tar(channel_dir: Path) -> Path:
    """If *channel_dir* contains a single tar.gz, extract and return the extracted path."""
    if not channel_dir.is_dir():
        return channel_dir
    tars = list(channel_dir.glob("*.tar.gz"))
    if not tars:
        return channel_dir
    dest = channel_dir / "_extracted"
    dest.mkdir(exist_ok=True)
    for t in tars:
        with tarfile.open(t, "r:gz") as tf:
            tf.extractall(dest)
    return dest


if __name__ == "__main__":
    mlm_channel = os.environ.get("SM_CHANNEL_MLM_CHECKPOINT", "")
    if mlm_channel and Path(mlm_channel).is_dir():
        extracted = _extract_channel_tar(Path(mlm_channel))
        os.environ["SM_CHANNEL_MLM_CHECKPOINT"] = str(extracted)
        import sys
        for i, arg in enumerate(sys.argv):
            if arg == "--mlm_checkpoint" and i + 1 < len(sys.argv):
                sys.argv[i + 1] = str(extracted)

    from training.train_classifier import main
    main()
