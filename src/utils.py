"""Shared helpers for the Cambodia Climate–Water–Energy project.

Notebooks import from here instead of copying functions:

    from src.utils import spi_1d, area_mean

Sections
--------
1. Drought statistics  : spi_1d, spi_series, yearly_value, dryness_rank
2. Spatial averaging   : area_mean, region_series
3. CHIRPS (IRI)        : iri_chirps, open_iri
4. CMIP6 (Pangeo)      : find_store, open_pr
5. HydroBASINS         : build_children, upstream_ids, snap_basin
"""

import re
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats


# ---------------------------------------------------------------------------
# 1. Drought statistics
# ---------------------------------------------------------------------------
def spi_1d(monthly_mm, months, window=3, cal_mask=None):
    """Standardized Precipitation Index for one monthly time series.

    Parameters
    ----------
    monthly_mm : 1-D array of monthly precipitation totals (mm/month), no gaps.
    months     : 1-D array of the calendar month (1–12) of each value.
    window     : accumulation period in months (3 = SPI-3, 12 = SPI-12).
    cal_mask   : boolean array, True for months used to fit the distribution
                 (the reference period). Default: all months.

    Method: rolling sum -> gamma fit per calendar month on the reference
    period (zero-inflated) -> standard normal quantile.
    The first ``window - 1`` values are NaN because the sum is incomplete.
    """
    monthly_mm = np.asarray(monthly_mm, dtype=float)
    months = np.asarray(months)
    if cal_mask is None:
        cal_mask = np.ones(monthly_mm.size, dtype=bool)
    cal_mask = np.asarray(cal_mask, dtype=bool)

    acc = pd.Series(monthly_mm).rolling(window, min_periods=window).sum().to_numpy()
    out = np.full_like(acc, np.nan, dtype=float)
    for m in range(1, 13):
        sel = months == m
        ref = acc[sel & cal_mask]
        ref = ref[~np.isnan(ref)]
        q = np.mean(ref <= 0)                                   # probability of zero
        a, _, scale = stats.gamma.fit(ref[ref > 0], floc=0)
        x = acc[sel]
        H = q + (1 - q) * stats.gamma.cdf(np.where(x > 0, x, 0), a, loc=0, scale=scale)
        H = np.where(x <= 0, q, H)
        out[sel] = stats.norm.ppf(np.clip(H, 1e-6, 1 - 1e-6))
    out[np.isnan(acc)] = np.nan                                 # week-2 bug fix
    return out


def spi_series(pr, window=3, base=("1985", "2014")):
    """SPI for a monthly pandas Series with a DatetimeIndex (mm/month)."""
    years = pr.index.year
    cal = (years >= int(base[0])) & (years <= int(base[1]))
    return pd.Series(spi_1d(pr.values, pr.index.month, window, cal), index=pr.index, name=f"SPI-{window}")


def yearly_value(s, month):
    """Pick one calendar month from a monthly Series -> Series indexed by year."""
    sub = s[s.index.month == month]
    return pd.Series(sub.values, index=sub.index.year).dropna()


def dryness_rank(s, year):
    """Rank of ``year`` in a yearly Series, 1 = driest (lowest). Returns (rank, n)."""
    return int(s.rank(method="min").loc[year]), int(s.notna().sum())


# ---------------------------------------------------------------------------
# 2. Spatial averaging
# ---------------------------------------------------------------------------
def area_mean(da):
    """Latitude-weighted (cos lat) mean over ``lat`` and ``lon``; NaNs are skipped."""
    w = np.cos(np.deg2rad(da.lat))
    return da.weighted(w).mean(("lat", "lon"))


def region_series(da, geom):
    """Area mean of ``da`` inside a shapely polygon, as a pandas Series.

    If the polygon is smaller than one grid cell, the nearest cell is used.
    """
    import regionmask

    m = regionmask.Regions([geom]).mask(da)
    if int(m.notnull().sum()) == 0:
        c = geom.centroid
        return da.sel(lon=c.x, lat=c.y, method="nearest").to_series()
    return area_mean(da.where(m.notnull())).to_series()


# ---------------------------------------------------------------------------
# 3. CHIRPS via the IRI Data Library
# ---------------------------------------------------------------------------
IRI_CHIRPS = {
    "v2p0": "https://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation",
    "v3p0": "https://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v3p0/.monthly/.global/.precipitation",
}


def iri_chirps(fn, bbox, versions=("v2p0", "v3p0"), timeout=900):
    """Download monthly CHIRPS for bbox = (west, south, east, north) to netCDF ``fn``.

    Returns (version, size_MB). Tries v2.0 first, then v3.0.
    """
    import requests

    x0, y0, x1, y1 = bbox
    for v in versions:
        url = f"{IRI_CHIRPS[v]}/X/%28{x0}%29%28{x1}%29RANGEEDGES/Y/%28{y1}%29%28{y0}%29RANGEEDGES/data.nc"
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            with open(fn, "wb") as f:
                f.write(r.content)
            xr.open_dataset(fn, decode_times=False).close()     # is it valid netCDF?
            return v, len(r.content) / 1e6
        except Exception as e:                                   # noqa: BLE001
            print(f"CHIRPS {v} failed: {type(e).__name__}")
    raise RuntimeError("CHIRPS download failed for all versions")


def open_iri(fn):
    """Open an IRI netCDF: rename X/Y/T, decode 'months since', return mm/month."""
    ds = xr.open_dataset(fn, decode_times=False)
    var = list(ds.data_vars)[0]
    da = ds[var].rename({k: v for k, v in {"X": "lon", "Y": "lat", "T": "time"}.items() if k in ds[var].dims})
    tunits = ds["T"].attrs.get("units", "") if "T" in ds else ""
    m = re.match(r"months since (\d{4})-(\d{1,2})", tunits)
    if m:
        t0 = pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
        dates = [t0 + pd.DateOffset(months=int(np.floor(v))) for v in da.time.values]
        da = da.assign_coords(time=pd.to_datetime(dates))
    if "day" in da.attrs.get("units", "").lower():
        da = da * da.time.dt.days_in_month
    return da.sortby("lat")


# ---------------------------------------------------------------------------
# 4. CMIP6 via the Pangeo Google Cloud catalogue
# ---------------------------------------------------------------------------
PANGEO_CATALOG = "https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv"


def find_store(cat, model, exp, table, var="pr"):
    """Pick one catalogue row: prefer member r1i1p1f1, native grid (gn), latest version."""
    sub = cat[(cat.source_id == model) & (cat.experiment_id == exp)
              & (cat.table_id == table) & (cat.variable_id == var)]
    if sub.empty:
        return None
    sub = sub.assign(_m=(sub.member_id != "r1i1p1f1").astype(int),
                     _g=(sub.grid_label != "gn").astype(int))
    sub = sub.sort_values(["_m", "_g", "member_id", "version"], ascending=[True, True, True, False])
    return sub.iloc[0]


def open_pr(row, period, lon, lat, margin=0.75, gcs=None, var="pr"):
    """Load a CMIP6 variable for a lon/lat box and period (tuple of year strings)."""
    if gcs is None:
        import gcsfs
        gcs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(gcs.get_mapper(row.zstore), consolidated=True)
    ds = ds.rename({k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items() if k in ds.dims})
    ds = ds.sortby("lat")
    if float(ds.lon.min()) < 0:
        ds = ds.assign_coords(lon=(ds.lon + 360) % 360).sortby("lon")
    return ds[var].sel(lat=slice(lat[0] - margin, lat[1] + margin),
                       lon=slice(lon[0] - margin, lon[1] + margin),
                       time=slice(*period)).load()


# ---------------------------------------------------------------------------
# 5. HydroBASINS upstream tracing
# ---------------------------------------------------------------------------
def build_children(hybas_ids, next_down):
    """Map each basin ID to the list of basins that drain directly into it."""
    children = defaultdict(list)
    for h, nd in zip(hybas_ids, next_down):
        children[nd].append(h)
    return children


def upstream_ids(children, start):
    """All basins upstream of ``start`` (inclusive), by breadth-first search."""
    seen, queue = {start}, deque([start])
    while queue:
        h = queue.popleft()
        for c in children.get(h, []):
            if c not in seen:
                seen.add(c)
                queue.append(c)
    return seen


def snap_basin(hb, lon, lat, radius_deg=0.03):
    """Basin within ``radius_deg`` of the point with the largest upstream area."""
    from shapely.geometry import Point

    cand = hb[hb.intersects(Point(lon, lat).buffer(radius_deg))]
    if cand.empty:
        raise ValueError(f"No basin within {radius_deg} deg of ({lon}, {lat})")
    return cand.loc[cand.UP_AREA.idxmax()]
