# Cambodia Climate-Water-Energy Risk

In March-April 2019, drought reduced Cambodia's hydropower output and led nationwide power cuts. This project asks how rainfall-driven risks to water and energy in Cambodia may change under climate change, and whether such droughts can be anticipated early enough to act.

**Approach:** CMIP6 climate scenarios -> observed drought analysis (CHIRPS) -> hydropower and flood exposure -> seasonal early warning.

## Progress

| Week | Notebook | Key findings |
| --- | --- | --- |
| 1 | [01_cmip6_starter](notebooks/01_cmip6_starter.ipynb) | MIROC6 (SSP2-4.5): annual rainfall +10% by 2041–2070, but March–April rainfall barely changes. Only heavy-rain days (R20mm) change significantly. |
| 2 | [01_cmip6_starter](notebooks/01_cmip6_starter.ipynb) §7 | Two models agree on more heavy-rain days and slightly drier March–April; they disagree on wet-season increase and absolute extremes → need multi-model analysis and bias correction. |
| 3 | [02_chirps_2019](notebooks/02_chirps_2019.ipynb) | Rainfall before the 2019 power cuts was only mildly below normal (12-month SPI -0.64, 14th driest of 45 years), but the 2019 wet season was the 2nd driest since 1981. El Nino strongly dries the dry season (r = -0.71). |
| 4 | [03_dam_catchments](notebooks/03_dam_catchments.ipynb) | Dam catchments were not unusually dry before the 2019 power cuts (capacity-weighted SPI-12 −0.58, 13th of 45), supporting a demand/vulnerability explanation. The real hydrological crisis was the 2020 dry season: Lower Sesan 2's mostly Vietnamese catchment was the driest since 1981. |
## Data
- CMIP6 daily and monthly precipitation (Pangeo Google Cloud catalogue)
- CHIRPS v2.0 monthly precipitation (IRI Data Library)

## Tools
Python · xarray · xclim · cartopy · scipy
