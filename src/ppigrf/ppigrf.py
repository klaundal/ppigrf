""" 
MIT License

Copyright (c) 2021 Karl M. Laundal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Pure Python implementation of IGRF. 


Example usage:
--------------
import numpy as np
from datetime import datetime
import ppigrf

# GEODETIC
lon = 5.32415  # degrees east
lat = 60.39299 # degrees north
h   = 0        # kilometers above sea level
date = datetime(2021, 3, 28)
Be, Bn, Bu = ppigrf.igrf(lon, lat, h, date) # returns east, north, up

# GEOCENTRIC
r     = 6500 # kilometers from center of Earht
theta = 30   # colatitude in degrees
phi   = 4    # degrees east (same as lon)
Br, Btheta, Bphi = ppigrf.igrf_gc(r, theta, phi, date) # returns radial, south, east

# GRID
lon = np.array([20, 120, 220])
lat = np.array([[60, 60, 60], [-60, -60, -60]])
h   = 0
dates = [datetime(y, 1, 1) for y in np.arange(1960, 2021, 20)]
Be, Bn, Bu = ppigrf.igrf(lon, lat, h, dates)



This script contains all code needed to calculate IGRF model values.
Here is a list of functions: 

Helper functions
----------------
is_leapyear          - Check if year is leapyear
yearfrac_to_datetime - Convert fraction of year to datetime
get_legendre         - Calculate Legendre functions
read_shc             - Read spherical harmonic coefficient file
geod2geoc            - Convert from geodetic to geocentric coordinates
geoc2geod            - Convert from geocentric to geodetic coordinates

Main functions
--------------
igrf_gc              - Calculate IGRF model values (geocentric input/output)
igrf                 - Calculate IGRF model values (geodetic input/output)

"""

import numpy as np
import pandas as pd
import os

basepath = os.path.dirname(__file__)
shc_fn = basepath + '/IGRF13.shc' # Default shc file

# Geomagnetic reference radius:
RE = 6371.2 # km

# World Geodetic System 84 parameters:
WGS84_e2 = 0.00669437999014
WGS84_a  = 6378.137 # km



def is_leapyear(year):
    """ Check for leapyear (handles arrays and preserves shape)

    """

    # if array:
    if type(year) is np.ndarray:
        out = np.full_like(year, False, dtype = bool)

        out[ year % 4   == 0] = True
        out[ year % 100 == 0] = False
        out[ year % 400 == 0] = True

        return out

    # if scalar:
    if year % 400 == 0:
        return True

    if year % 100 == 0:
        return False

    if year % 4 == 0:
        return True

    else:
        return False


def yearfrac_to_datetime(fracyear):
    """ 
    Convert fraction of year to datetime 

    Parameters
    ----------
    fracyear : iterable
        Date(s) in decimal year. E.g., 2021-03-28 is 2021.2377
        Must be an array, list or similar.

    Returns
    -------
    datetimes : array
        Array of datetimes
    """

    year = np.uint16(fracyear) # truncate fracyear to get year
    # use pandas to_timedelta to represent time since beginning of year: 
    delta_year = pd.to_timedelta((fracyear - year)*(365 + is_leapyear(year)), unit = 'D')
    # and DatetimeIndex to represent beginning of years:
    start_year = pd.DatetimeIndex(list(map(str, year)))
 
    # adding them produces the datetime:
    return (start_year + delta_year).to_pydatetime()


def get_legendre(theta, keys):
    """ 
    Calculate Schmidt semi-normalized associated Legendre functions

    Calculations based on recursive algorithm found in "Spacecraft Attitude Determination and Control" by James Richard Wertz
    
    Parameters
    ----------
    theta : array
        Array of colatitudes in degrees
    keys: iterable
        list of spherical harmnoic degree and order, tuple (n, m) for each 
        term in the expansion

    Returns
    -------
    P : array 
        Array of Legendre functions, with shape (theta.size, len(keys)). 
    dP : array
        Array of dP/dtheta, with shape (theta.size, len(keys))
    """

    # get maximum N and maximum M:
    n, m = np.array([k for k in keys]).T
    nmax, mmax = np.max(n), np.max(m)

    theta = theta.flatten()[:, np.newaxis]
    theta_rad = np.radians(theta)

    P = {}
    dP = {}
    sinth = np.sin(theta_rad)
    costh = np.cos(theta_rad)

    # Initialize Schmidt normalization
    S = {}
    S[0, 0] = 1.

    # initialize the functions:
    for n in range(nmax +1):
        for m in range(nmax + 1):
            P[n, m] = np.zeros_like(theta, dtype = np.float64)
            dP[n, m] = np.zeros_like(theta, dtype = np.float64)

    P[0, 0] = np.ones_like(theta, dtype = np.float64)
    for n in range(1, nmax +1):
        for m in range(0, min([n + 1, mmax + 1])):
            # do the legendre polynomials and derivatives
            if n == m:
                P[n, n]  = sinth * P[n - 1, m - 1]
                dP[n, n] = sinth * dP[n - 1, m - 1] + costh * P[n - 1, n - 1]
            else:

                if n == 1:
                    Knm = 0.
                    P[n, m]  = costh * P[n -1, m]
                    dP[n, m] = costh * dP[n - 1, m] - sinth * P[n - 1, m]

                elif n > 1:
                    Knm = ((n - 1)**2 - m**2) / ((2*n - 1)*(2*n - 3))
                    P[n, m]  = costh * P[n -1, m] - Knm*P[n - 2, m]
                    dP[n, m] = costh * dP[n - 1, m] - sinth * P[n - 1, m] - Knm * dP[n - 2, m]

            # compute Schmidt normalization
            if m == 0:
                S[n, 0] = S[n - 1, 0] * (2.*n - 1)/n
            else:
                S[n, m] = S[n, m - 1] * np.sqrt((n - m + 1)*(int(m == 1) + 1.)/(n + m))


    # now apply Schmidt normalization
    for n in range(1, nmax + 1):
        for m in range(0, min([n + 1, mmax + 1])):
            P[n, m]  *= S[n, m]
            dP[n, m] *= S[n, m]

    Pmat  = np.hstack(tuple(P[key] for key in keys))
    dPmat = np.hstack(tuple(dP[key] for key in keys)) 
    return Pmat, dPmat    


def read_shc(filename = shc_fn):
    """ 
    Read .shc (spherical harmonic coefficient) file

    The function produces data frames that have time as index and 
    spherical harmonic degree and order as columns. In the case of IGRF,
    the times will correspond to the different models 5 years apart

    Parameters
    ----------
    filename : string
        filename of .shc file

    Returns
    -------
    g : DataFrame
        pandas DataFrame of gauss coefficients for cos terms.
    h : DataFrame
        pandas DataFrame of gauss coefficients for sin terms.

    Note
    ----
    This code has no special treatment of "secular variation" coefficients. 
    Instead, the SV coefficients of IGRF should be used to make gauss 
    coefficients. This must be done prior to this code (when making the 
    .shc file).
    """

    header = 2
    coeffdict = {}
    with open(filename, 'r') as f:
        for line in f.readlines():

            if line.startswith('#'): # this is a header that we don't read
                continue

            if header == 2: # read parameters (could be skipped...)
                 N_MIN, N_MAX, NTIMES, SP_ORDER, N_STEPS = list(map(int, line.split()[:5]))
                 header -= 1
                 continue

            if header == 1: # read years
                 times = yearfrac_to_datetime(list(map(float, line.split())) )
                 header -= 1
                 continue

            key = tuple(map(int, line.split()[:2]))
            coeffdict[key] = np.array(list(map(float, line.split()[2:])))

    g = {key:coeffdict[key] for key in coeffdict.keys() if key[1] >= 0}
    h = {(key[0], -key[1]):coeffdict[key] for key in coeffdict.keys() if key[1] < 0 }
    for key in [k for k in g.keys() if k[1] == 0]: # add zero coefficients for m = 0 in h dictionary
        h[key] = 0

    # this must be true:
    assert len(g.keys()) == len(h.keys())

    gdf = pd.DataFrame(g, index = times)
    hdf = pd.DataFrame(h, index = times)

    # make sure that the column keys of h are in same order as in g:
    hdf = hdf[gdf.columns]

    return gdf, hdf



def geod2geoc(gdlat, height, Bn, Bu):
    """
    Convert from geocentric to geodetic coordinates

    Example:
    --------
    theta, r, B_th, B_r = geod2lat(gdlat, height, Bn, Bu)

    Parameters
    ----------
    gdlat : array
        Geodetic latitude [degrees]
    h : array
        Height above ellipsoid [km]
    Bn : array
        Vector in northward direction, relative to ellipsoid
    Bu : array
        Vector in upward direction, relative to ellipsoid

    Returns
    -------
    theta : array
        Colatitudes [degrees]
    r : array
        Radius [km]
    B_th : array
        Vector component in theta direction
    B_r : array
        Vector component in radial direction
    """

    a = WGS84_a
    b = a*np.sqrt(1 - WGS84_e2)

    # Convert geodetic latitude angles to radians
    gdlat_rad = np.radians(gdlat)

    sin_alpha_2 = np.sin(gdlat_rad)**2
    cos_alpha_2 = np.cos(gdlat_rad)**2

    # calculate geocentric latitude and radius
    tmp = height * np.sqrt(a**2 * cos_alpha_2 + b**2 * sin_alpha_2)
    beta = np.arctan((tmp + b**2)/(tmp + a**2) * np.tan(gdlat_rad))
    theta = np.pi/2 - beta
    r = np.sqrt(height**2 + 2 * tmp + a**2 * (1 - (1 - (b/a)**4) * sin_alpha_2) / (1 - (1 - (b/a)**2) * sin_alpha_2))

    # calculate geocentric components
    psi  =  np.sin(gdlat_rad) * np.sin(theta) - np.cos(gdlat_rad) * np.cos(theta)
    
    B_r  = -np.sin(psi) * Bn + np.cos(psi) * Bu
    B_th = -np.cos(psi) * Bn - np.sin(psi) * Bu

    # Convert theta to degrees
    theta = np.degrees(theta)

    return theta, r, B_th, B_r
 

def geoc2geod(theta, r, B_th, B_r):
    """
    Convert from geodetic to geocentric coordinates

    Based on Matlab code by Nils Olsen, DTU

    Example:
    --------
    gdlat, height, Bn, Bu = geod2lat(theta, r, B_th, B_r)

    Parameters
    ----------
    theta : array
        Colatitudes [degrees]
    r : array
        Radius [km]
    B_th : array
        Vector component in theta direction
    B_r : array
        Vector component in radial direction

    Returns
    -------
    gdlat : array
        Geodetic latitude [degrees]
    h : array
        Height above ellipsoid [km]
    Bn : array
        Vector in northward direction, relative to ellipsoid
    Bu : array
        Vector in upward direction, relative to ellipsoid
    """
    
    a = WGS84_a
    b = a*np.sqrt(1 - WGS84_e2)

    E2 = 1.-(b/a)**2
    E4 = E2*E2
    E6 = E4*E2
    E8 = E4*E4
    OME2REQ = (1.-E2)*a
    A21 =     (512.*E2 + 128.*E4 + 60.*E6 + 35.*E8)/1024.
    A22 =     (                        E6 +     E8)/  32.
    A23 = -3.*(                     4.*E6 +  3.*E8)/ 256.
    A41 =    -(           64.*E4 + 48.*E6 + 35.*E8)/1024.
    A42 =     (            4.*E4 +  2.*E6 +     E8)/  16.
    A43 =                                   15.*E8 / 256.
    A44 =                                      -E8 /  16.
    A61 =  3.*(                     4.*E6 +  5.*E8)/1024.
    A62 = -3.*(                        E6 +     E8)/  32.
    A63 = 35.*(                     4.*E6 +  3.*E8)/ 768.
    A81 =                                   -5.*E8 /2048.
    A82 =                                   64.*E8 /2048.
    A83 =                                 -252.*E8 /2048.
    A84 =                                  320.*E8 /2048.
    
    GCLAT = (90-theta)
    SCL = np.sin(np.radians(GCLAT))
    
    RI = a/r
    A2 = RI*(A21 + RI * (A22 + RI* A23))
    A4 = RI*(A41 + RI * (A42 + RI*(A43+RI*A44)))
    A6 = RI*(A61 + RI * (A62 + RI* A63))
    A8 = RI*(A81 + RI * (A82 + RI*(A83+RI*A84)))
    
    CCL = np.sqrt(1-SCL**2)
    S2CL = 2.*SCL  * CCL
    C2CL = 2.*CCL  * CCL-1.
    S4CL = 2.*S2CL * C2CL
    C4CL = 2.*C2CL * C2CL-1.
    S8CL = 2.*S4CL * C4CL
    S6CL = S2CL * C4CL + C2CL * S4CL
    
    DLTCL = S2CL * A2 + S4CL * A4 + S6CL * A6 + S8CL * A8
    gdlat = DLTCL + np.radians(GCLAT)
    height = r * np.cos(DLTCL)- a * np.sqrt(1 -  E2 * np.sin(gdlat) ** 2)


    # magnetic components 
    theta_rad = np.radians(theta)
    psi = np.sin(gdlat) * np.sin(theta_rad) - np.cos(gdlat) * np.cos(theta_rad)
    Bn = -np.cos(psi) * B_th - np.sin(psi) * B_r 
    Bu = -np.sin(psi) * B_th + np.cos(psi) * B_r 

    # Convert gdlat to degrees
    gdlat = np.degrees(gdlat)

    return gdlat, height, Bn, Bu



def igrf_gc(r, theta, phi, date, coeff_fn = shc_fn):
    """
    Calculate IGRF model components

    Input and output in geocentric coordinates

    Broadcasting rules apply for coordinate arrays, and the
    combined shape will be preserved. The dates are kept out
    of the broadcasting, so that the output will have shape
    (N, ...) where N is the number of dates, and ... represents
    the combined shape of the coordinates. If you pass scalars,
    the output will be arrays of shape (1,)
    
    Parameters
    ----------
    r : array
        radius [km] of IGRF calculation
    theta : array
        colatitude [deg] of IGRF calculation
    phi : array
        longitude [deg], positive east, of IGRF claculation
    date : date(s)
        one or more dates to evaluate IGRF coefficients
    coeff_fn : string, optional
        filename of .shc file. Default is latest IGRF

    Return
    ------
    Br : array
        Magnetic field [nT] in radial direction
    Btheta : array
        Magnetic field [nT] in theta direction (south on an
        Earth-centered sphere with radius r)
    Bphi : array
        Magnetic field [nT] in eastward direction
    """

    # read coefficient file:
    g, h = read_shc(coeff_fn)

    if not hasattr(date, '__iter__'):
        date = np.array([date])
    else:
        date = np.array(date)

    if np.any(date > g.index[-1]) or np.any(date < g.index[0]):
        print('Warning: You provided date(s) not covered by coefficient file \n({} to {})'.format(
              g.index[0].date(), g.index[-1].date()))

    # get coordinate arrays to same size and shape
    r, theta, phi = np.broadcast_arrays(r, theta, phi)
    shape = r.shape
    r, theta, phi = map(lambda x: x.flatten().reshape((-1 ,1)), [r, theta, phi]) # column vectors

    # make row vectors of wave numbers n and m:
    n, m = np.array([k for k in g.columns]).T
    n, m = n.reshape((1, -1)), m.reshape((1, -1))

    # get maximum N and maximum M:
    N, M = np.max(n), np.max(m)

    # get the legendre functions
    P, dP = get_legendre(theta, g.keys())

    # Append coefficients at desired times (skip if index is already in coefficient data frame):
    index = g.index.union(date)

    g = g.reindex(index).groupby(index).first() # reindex and skip duplicates
    h = h.reindex(index).groupby(index).first() # reindex and skip duplicates

    # interpolate and collect the coefficients at desired times:
    g = g.interpolate(method = 'time').loc[date, :]
    h = h.interpolate(method = 'time').loc[date, :]

    # compute cosmlon and sinmlon:
    phi_rad = np.radians(phi)
    cosmphi = np.cos(phi_rad * m) # shape (n_coords x n_model_params/2)
    sinmphi = np.sin(phi_rad * m)

    # make versions of n and m that are repeated twice
    nn, mm = np.tile(n, 2), np.tile(m, 2)

    # calculate Br:
    G  = (RE / r) ** (nn + 2) * (nn + 1) * np.hstack((P * cosmphi, P * sinmphi))
    Br = G.dot(np.hstack((g.values, h.values)).T).T # shape (n_times, n_coords)

    # calculate Btheta:
    G  = -(RE / r) ** (nn + 1) * np.hstack((dP * cosmphi, dP * sinmphi)) \
         * RE / r
    Btheta = G.dot(np.hstack((g.values, h.values)).T).T # shape (n_times, n_coords)

    # calculate Bphi:
    G  = -(RE / r) ** (nn + 1) * mm * np.hstack((-P * sinmphi, P * cosmphi)) \
         * RE / r / np.sin(np.radians(theta))
    Bphi = G.dot(np.hstack((g.values, h.values)).T).T # shape (n_times, n_coords)

    # reshape and return
    outshape = tuple([Bphi.shape[0]] + list(shape))
    return Br.reshape(outshape), Btheta.reshape(outshape), Bphi.reshape(outshape)


def igrf(lon, lat, h, date, coeff_fn = shc_fn):
    """
    Calculate IGRF model components

    Input and output in geodetic coordinates

    Broadcasting rules apply for coordinate arrays, and the
    combined shape will be preserved. The dates are kept out
    of the broadcasting, so that the output will have shape
    (N, ...) where N is the number of dates, and ... represents
    the combined shape of the coordinates. If you pass scalars,
    the output will be arrays of shape (1,)
    
    Parameters
    ----------
    lon : array
        longitude [deg], postiive east, of IGRF calculation
    lat : array
        geodetic latitude [deg] of IGRF calculation
    h : array
        height [km] above ellipsoid for IGRF calculation
    date : date(s)
        one or more dates to evaluate IGRF coefficients
    coeff_fn : string, optional
        filename of .shc file. Default is latest IGRF

    Return
    ------
    Be : array
        Magnetic field [nT] in eastward direction
    Bn : array
        Magnetic field [nT] in northward direction, relative to
        ellipsoid
    Bu : array
        Magnetic field [nT] in upward direction, relative to 
        ellipsoid
    """

    # convert input to arrays and cast to same shape:
    lon, lat, h = np.broadcast_arrays(lon, lat, h)
    shape = lon.shape
    lon, lat, h = map(lambda x: x.flatten(), [lon, lat, h])

    # convert to geocentric:
    theta, r, _, __ = geod2geoc(lat, h, h, h)
    phi = lon

    # calculate geocentric components of IGRF:
    Br, Btheta, Bphi = igrf_gc(r, theta, phi, date, coeff_fn = coeff_fn)
    Be = Bphi

    # convert output to geodetic
    lat_, h_, Bn, Bu = geoc2geod(theta, r, Btheta, Br)

    # return with shapes implied by input
    outshape = tuple([Be.shape[0]] + list(shape))
    return Be.reshape(outshape), Bn.reshape(outshape), Bu.reshape(outshape)


def igrf_V(r, theta, phi, date, coeff_fn = shc_fn):
    """
    Calculate IGRF magnetic potential

    Input and output in geocentric coordinates

    Broadcasting rules apply for coordinate arrays, and the
    combined shape will be preserved. The dates are kept out
    of the broadcasting, so that the output will have shape
    (N, ...) where N is the number of dates, and ... represents
    the combined shape of the coordinates. If you pass scalars,
    the output will be arrays of shape (1,)
    
    Parameters
    ----------
    r : array
        radius [km] of IGRF calculation
    theta : array
        colatitude [deg] of IGRF calculation
    phi : array
        longitude [deg], positive east, of IGRF claculation
    date : date(s)
        one or more dates to evaluate IGRF coefficients
    coeff_fn : string, optional
        filename of .shc file. Default is latest IGRF

    Return
    ------
    V : array
        Magnetic potential (unit nT * km)
    """

    # read coefficient file:
    g, h = read_shc(coeff_fn)

    if not hasattr(date, '__iter__'):
        date = np.array([date])
    else:
        date = np.array(date)

    if np.any(date > g.index[-1]) or np.any(date < g.index[0]):
        print('Warning: You provided date(s) not covered by coefficient file \n({} to {})'.format(
              g.index[0].date(), g.index[-1].date()))

    # convert input to arrays in case they aren't
    r, theta, phi = np.broadcast_arrays(r, theta, phi)
    shape = r.shape
    r, theta, phi = map(lambda x: x.flatten().reshape((-1 ,1)), [r, theta, phi]) # column vectors

    # make row vectors of wave numbers n and m:
    n, m = np.array([k for k in g.columns]).T
    n, m = n.reshape((1, -1)), m.reshape((1, -1))

    # get maximum N and maximum M:
    N, M = np.max(n), np.max(m)

    # get the legendre functions
    P, dP = get_legendre(theta, g.keys())

    # Append coefficients at desired times (skip if index is already in coefficient data frame):
    index = g.index.union(date)

    g = g.reindex(index).groupby(index).first() # reindex and skip duplicates
    h = h.reindex(index).groupby(index).first() # reindex and skip duplicates

    # interpolate and collect the coefficients at desired times:
    g = g.interpolate(method = 'time').loc[date, :]
    h = h.interpolate(method = 'time').loc[date, :]

    # compute cosmlon and sinmlon:
    cosmphi = np.cos(np.deg2rad(phi * m)) # shape (n_coords x n_model_params/2)
    sinmphi = np.sin(np.deg2rad(phi * m))

    # make versions of n and m that are repeated twice
    nn, mm = np.tile(n, 2), np.tile(m, 2)

    # calculate Br:
    G  = RE * (RE / r) ** (nn + 1) * np.hstack((P * cosmphi, P * sinmphi))
    V = G.dot(np.hstack((g.values, h.values)).T).T # shape (n_times, n_coords)

    # reshape and return
    outshape = tuple([V.shape[0]] + list(shape))
    return V.reshape(outshape)


def get_inclination_declination(Be, Bn, Bu, degrees=True):
    r"""
    Compute the inclination and declination angles of the IGRF

    The inclination angle is defined as the angle between the magnetic field
    vector and the horizontal plane:

    .. math::

        I = \arctan \frac{-B_u}{\sqrt{B_e^2 + B_n^2}}

    And the declination angle is defined as the azimuth of the projection of
    the magnetic field vector onto the horizontal plane (starting from the
    northing direction, positive to the east and negative to the west):

    .. math::

        D = \arcsin \frac{B_e}{\sqrt{B_e^2 + B_n^2}}

    Parameters
    ----------
    Be : float or array
        Easting component of the IGRF magnetic vector.
    Bn : float or array
        Northing component of the IGRF magnetic vector.
    Bu : float or array
        Upward component of the IGRF magnetic vector.
    degrees : bool (optional)
        If True, the angles are returned in degrees.
        If False, the angles are returned in radians.
        Default True.

    Returns
    -------
    inclination : float or array
        Inclination angle of the IGRF magnetic vector. If ``degrees`` is True,
        then the angle is returned in degrees. If ``degrees`` is False, then
        it's returned in radians.
    declination : float or array
        Declination angle of the IGRF magnetic vector. If ``degrees`` is True,
        then the angle is returned in degrees. If ``degrees`` is False, then
        it's returned in radians.
    """
    # Compute the horizontal component of B
    horizontal_component = np.sqrt(Be**2 + Bn**2)
    if horizontal_component == 0:
        inclination = -np.sign(Bu) * np.pi /2
        declination = 0
    else:
        # Compute the two angles
        inclination = np.arctan(-Bu / horizontal_component)
        declination = np.arcsin(Be / horizontal_component)
    # Convert to degrees if needed
    if degrees:
        inclination = np.degrees(inclination)
        declination = np.degrees(declination)
    return inclination, declination


if __name__ == '__main__':

    from datetime import datetime

    # GEODETIC
    lon = 5.32415  # degrees east
    lat = 60.39299 # degrees north
    h   = 0        # kilometers above sea level
    date = datetime(2021, 3, 28)
    Be, Bn, Bu = igrf(lon, lat, h, date) # returns east, north, up

    # GEOCENTRIC
    r     = 6500 # kilometers from center of Earht
    theta = 30   # colatitude in degrees
    phi   = 4    # degrees east (same as lon)
    Br, Btheta, Bphi = igrf_gc(r, theta, phi, date) # returns radial, south, east

    # GRID
    lon = np.array([20, 120, 220])
    lat = np.array([[60, 60, 60], [-60, -60, -60]])
    h   = 0
    dates = [datetime(y, 1, 1) for y in np.arange(1960, 2021, 20)]
    Be, Bn, Bu = igrf(lon, lat, h, dates)
