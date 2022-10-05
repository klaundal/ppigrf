# Data files used for testing ppigrf

## Precomputed IGRF

The `b_e.csv`, `b_n.csv` and the `b_z.csv` files contain precomputed values of
the IGRF field obtained through the [NCEI Geomagnetic
Calculator](https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml)
provided by [NOAA](https://www.ngdc.noaa.gov).

These values were computed in a regular grid with a spacing of 1 degree in both
longitudinal and latitudinal directions, at a height of 5km above the WGS84
ellipsoid and dated on October 5, 2022.

The `b_z.csv` contains the **downward** components of the magnetic vector on
each location.
