# Pure Python International Geomagnetic Reference Field
Pure Python code to calculate IGRF model predictions. The IGRF is a model of the Earth's main magnetic field that is updated every 5 years. 

Calculations are all vectorized, so calculations should be pretty fast.  

## How?
The only dependencies are Numpy and Pandas. There is no install. Just copy ppigrf.py and IGRF13.shc to your working directory and you are good to go. Or have the ppigrf directory somewhere that Python can find it. Then you should be able to import like this:
```python
import ppigrf
```
IGRF model values depend on time and position. Time can be given as one or more datetimes. The position can be specified in either geodetic or geocentric coordinates. Example with geodetic coordinates:
```python
from datetime import datetime
lon = 5.32415  # degrees east
lat = 60.39299 # degrees north
h   = 0        # kilometers above sea level
date = datetime(2021, 3, 28)

Be, Bn, Bu = ppgirf.igrf(lon, lat, h, date) # returns east, north, up
```
Geodetic coordinates take the ellipsoidal shape of the Earth into account. The northward component returned by the igrf function is tangential to the ellipsoid, and in general not tangential to an Earth centered sphere. The upward component is perpendicular to the ellipsoid, and in general not perpendicular to the sphere. 

In some cases it is more convenient to work in geocentric coordinates, which are purely spherical. To do that, use the igrf_gc function:
```python
r     = 6500 # kilometers from center of Earht
theta = 30   # colatitude in degrees
phi   = 4    # degrees east (same as lon)

Br, Btheta, Bphi = ppigrf.igrf_gc(r, theta, phi, date) # returns radial, south, east
```

It is also possible to calculate magnetic field values on a grid. The code uses broadcasting rules for the coordinate inputs. You can also pass multiple dates. If you pass `K` time stamps and coordinates with a combined shape of e.g., `(L, M, N)`, the output will have shape `(K, L, M, N)`. Example:
```python
lon = np.array([20, 120, 220])
lat = np.array([[60, 60, 60], [-60, -60, -60]])
h   = 0
dates = [datetime(y, 1, 1) for y in np.arange(1960, 2021, 20)]
Be, Bn, Bu = ppigrf.igrf(lon, lat, h, dates)
```
The output will have shape `(4, 2, 3)`.

## Why?
There are lots Python modules that can calculate IGRF values. Most are wrappers of Fortran code, which can be tricky to compile. This version is pure Python. I also prefer the ppigrf interface over the alternatives.


## Notes
The model coefficients are read from an .shc file. This is a file format that is used for certain spherical harmonic magnetic field models. See a selection of .shc model files here:
https://github.com/ESA-VirES/MagneticModel/blob/staging/eoxmagmod/eoxmagmod/data/

It should be straightforward to swap the IGRF .shc file with another model, but keep in mind that the time dependence may be implemented differently in other models. IGRF, and this code, uses linear interpolation between model updates, and changing the model file will not change the ppigrf interpolation setup. 


## Contact
If you find errors, please let me know! 

You don't need to ask permission to copy or use this code, but I would be happy to know if you do. 

karl.laundal at uib.no