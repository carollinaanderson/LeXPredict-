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
