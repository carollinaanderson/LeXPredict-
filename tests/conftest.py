import os

# Must run BEFORE any OpenMP-backed library imports (numpy, torch, lightgbm, shap).
# PyTorch bundles its own libomp; on macOS it collides with Homebrew's libomp that
# LightGBM/SHAP load, and the duplicate OpenMP runtime segfaults the suite at a random
# test. A single OpenMP thread makes the whole suite deterministic and avoids the crash.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest

from claimvalue.anonymize import anonymize_cases
from claimvalue.db import build_db
from claimvalue.synthetic import generate_raw_cases

PEPPER = "unit-test-pepper-" + "x" * 32


@pytest.fixture(scope="session")
def raw():
    return generate_raw_cases(n=1500, n_judges=20, seed=7)


@pytest.fixture(scope="session")
def anon(raw):
    return anonymize_cases(raw, PEPPER)


@pytest.fixture(scope="session")
def db_path(anon, tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "claims.db"
    build_db(anon, path)
    return path
