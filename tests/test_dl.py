"""Deep-learning test, isolated in a subprocess.

Why a subprocess (the non-obvious part): PyTorch bundles its own OpenMP runtime. When both
torch *and* LightGBM (Homebrew libomp on macOS) live in one process — which happens the
moment ``pytest`` collects this module and then runs ``test_training`` — the duplicate
OpenMP runtime segfaults the interpreter non-deterministically. ``importorskip("torch")``
at collection time already loads torch into the main process, so even ``pytest-forked``
cannot save it. Running the torch workload in a child process keeps the main process
torch-free while still asserting the DL pipeline works end to end.
"""

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

if importlib.util.find_spec("torch") is None:
    pytestmark = pytest.mark.skip(reason="torch not installed (pip install -e '.[dl]')")

_SRC = Path(__file__).resolve().parents[1] / "src"

_SCRIPT = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    from claimvalue.dl import train_dl_regression

    db, out = Path(sys.argv[1]), Path(sys.argv[2])
    res = train_dl_regression(db, out, epochs=3, patience=2)
    assert res["test_metrics"]["mae"] > 0 and res["epochs_run"] >= 1
    assert (out / "dl_regression.pt").exists()
    """
)


def test_dl_trains_and_reports_metrics(db_path, tmp_path):
    # PYTHONPATH makes the child independent of the editable-install .pth files, which macOS
    # can silently mark hidden. OMP_NUM_THREADS=1 keeps the single-threaded check cheap.
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [str(_SRC), os.environ.get("PYTHONPATH")])),
        "OMP_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(db_path), str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr[-2000:]
