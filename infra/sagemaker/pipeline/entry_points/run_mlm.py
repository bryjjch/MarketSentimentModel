"""Thin SageMaker entry-point wrapper for MLM continued pre-training.

The ``training`` package is copied into the source tarball via the estimator's
``dependencies`` parameter so that relative imports resolve normally.
"""

from training.train_mlm import main

if __name__ == "__main__":
    main()
