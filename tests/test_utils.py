"""Tests for src/utils.py. Run from the repository root:

    python -m pytest -q
"""

import numpy as np
import pandas as pd
import pytest

from src.utils import build_children, dryness_rank, spi_1d, spi_series, upstream_ids, yearly_value

# Cambodia-like monthly climatology (mm/month), dry Jan–Apr, wet May–Oct
CLIM = np.array([10, 15, 40, 80, 180, 200, 220, 240, 280, 220, 90, 25.0])


def synthetic(n_years=60, seed=0, future_factor=1.0, split_year=30):
    rng = np.random.default_rng(seed)
    months = np.tile(np.arange(1, 13), n_years)
    pr = rng.gamma(4, CLIM[months - 1] / 4)
    pr[np.arange(pr.size) >= 12 * split_year] *= future_factor
    base = np.arange(pr.size) < 12 * split_year
    return pr, months, base


@pytest.mark.parametrize("window", [3, 6, 12])
def test_spi_is_standard_normal_in_reference_period(window):
    pr, months, base = synthetic()
    s = spi_1d(pr, months, window, base)[base]
    s = s[~np.isnan(s)]
    assert abs(s.mean()) < 0.1
    assert 0.85 < s.std() < 1.15
    assert 0.10 < np.mean(s <= -1) < 0.22          # theory: 15.9%


@pytest.mark.parametrize("window", [3, 6, 12])
def test_spi_leading_values_are_nan(window):
    pr, months, base = synthetic()
    s = spi_1d(pr, months, window, base)
    assert np.isnan(s[: window - 1]).all()
    assert not np.isnan(s[window - 1])


def test_spi_detects_drier_future():
    pr, months, base = synthetic(future_factor=0.7)
    s = spi_1d(pr, months, 3, base)
    assert np.nanmean(s[~base]) < -0.5


def test_spi_zero_rain_floor_is_norm_ppf_q():
    # Half of all Januaries have zero rain -> SPI of a zero January is ppf(0.5) = 0
    pr, months, base = synthetic(n_years=40, split_year=40)
    jan = np.where(months == 1)[0]
    pr[jan[::2]] = 0.0
    s = spi_1d(pr, months, 1, base)
    assert np.allclose(s[jan[::2]], 0.0, atol=1e-9)


def test_spi_series_matches_spi_1d():
    pr, months, _ = synthetic()
    idx = pd.date_range("1981-01-01", periods=pr.size, freq="MS")
    s = spi_series(pd.Series(pr, index=idx), window=6, base=("1985", "2014"))
    base = (idx.year >= 1985) & (idx.year <= 2014)
    np.testing.assert_allclose(s.values, spi_1d(pr, idx.month, 6, base), equal_nan=True)


def test_yearly_value_and_dryness_rank():
    idx = pd.date_range("2000-01-01", periods=36, freq="MS")
    s = pd.Series(np.arange(36, dtype=float), index=idx)
    apr = yearly_value(s, 4)
    assert list(apr.index) == [2000, 2001, 2002]
    assert list(apr.values) == [3.0, 15.0, 27.0]
    assert dryness_rank(apr, 2000) == (1, 3)
    assert dryness_rank(apr, 2002) == (3, 3)


def test_upstream_tracing_on_toy_network():
    #   3 ─┐
    #      ├─> 2 ─> 1 ─> sea(0)        5 ─> sea(0)
    #   4 ─┘
    ids, nxt = [1, 2, 3, 4, 5], [0, 1, 2, 2, 0]
    ch = build_children(ids, nxt)
    assert upstream_ids(ch, 1) == {1, 2, 3, 4}
    assert upstream_ids(ch, 2) == {2, 3, 4}
    assert upstream_ids(ch, 3) == {3}
    assert upstream_ids(ch, 5) == {5}


def test_area_mean_weights_by_latitude():
    xr = pytest.importorskip("xarray")
    from src.utils import area_mean

    da = xr.DataArray([[1.0, 1.0], [3.0, 3.0]], coords={"lat": [0.0, 60.0], "lon": [100.0, 101.0]},
                      dims=("lat", "lon"))
    # weights cos(0)=1, cos(60)=0.5 -> (1*1 + 3*0.5) / 1.5 = 1.667
    assert float(area_mean(da)) == pytest.approx(5 / 3)
