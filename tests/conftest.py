import pytest

from toxion.data.synthetic import make_synthetic_clean


@pytest.fixture(scope="session")
def synthetic_df():
    return make_synthetic_clean(n=150, seed=0)
