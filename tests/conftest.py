import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))  # make _helpers importable from tests

from _helpers import build_growth_fig  # noqa: E402


@pytest.fixture
def growth_fig():
    fig = build_growth_fig()
    yield fig
    plt.close(fig)
