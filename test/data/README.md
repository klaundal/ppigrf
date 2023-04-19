# Data files used for testing ppigrf

## Precomputed IGRF

Each one of these folders contain a set of `b_e.csv`, `b_n.csv` and `b_z.csv`
files that host precomputed values of the IGRF field obtained through the [NCEI
Geomagnetic
Calculator](https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml)
provided by [NOAA](https://www.ngdc.noaa.gov).

These values were computed in a regular grid with a spacing of 1 degree in both
longitudinal and latitudinal directions, at a height of 5km above the WGS84
ellipsoid. The date of each IGRF field files is specified in the folder name
following the `YYYY-MM-DD` format.

The `b_z.csv` contains the **downward** components of the magnetic vector on
each location.
