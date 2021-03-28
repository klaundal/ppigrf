""" Script to calcualte IGRF 

Pure Python implementation of IGRF. 

The code should work with any main field model specified by coefficient in a shc file,
assuming linear interpolation between model times. 

https://github.com/ESA-VirES/MagneticModel/blob/staging/eoxmagmod/eoxmagmod/data/


Example usage:
--------------
from igrf import igrf_gc, igrf

Br, Btheta, Bphi = igrf_gc(r, theta, phi, dates) # input in geocentric
Be, Bn, Bu       = igrf(lon, lat, h, dates) # input in geodetic coordinates




Helper functions are defined first, and then the main IGRF function

Helper functions
----------------
get_legendre         - Calculate Legendre functions with a recursive algorithm found in 
                       "Spacecraft Attitude Determination and Control" by James Richard Wertz 
read_shc             - Read spherical harmonic coefficient file (shc format) and return gauss
                       coefficients as pandas DataFrame 
yearfrac_to_datetime - Convert fraction of year to datetime
is_leapyear          - Check if year is leapyear
geod2geoc            - Convert from geodetic to geocentric coordinates
geoc2geod            - Convert from geocentric to geodetic coordinates

"""

import numpy as np
import pandas as pd
import os
d2r = np.pi/180

basepath = os.path.dirname(__file__)
shc_fn = basepath + '/IGRF13.shc' # Default shc file

# Geomagnetic reference radius:
REFRE = 6371.2 

# World Geodetic System 84 parameters:
WGS84_e2 = 0.00669437999014
WGS84_a  = 6378.137 # km



def is_leapyear(year):
    """ Return True if leapyear else False
    
        handles arrays (preserves shape).

        KML 2016-04-20
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


def yearfrac_to_datetime(number):
    """ convert fraction of year to datetime 

        handles arrays


        example:

        >>> dates.yearfrac_to_datetime(np.array([2000.213]))
        array([datetime.datetime(2000, 3, 18, 22, 59, 31, 200000)], dtype=object)
    """
    
    year = np.uint16(number) # truncate number to get year
    # use pandas TimedeltaIndex to represent time since beginning of year: 
    delta_year = pd.TimedeltaIndex((number - year)*(365 + is_leapyear(year)), unit = 'D')
    # and DatetimeIndex to represent beginning of years:
    start_year = pd.DatetimeIndex(list(map(str, year)))
 
    # adding them produces the datetime:
    return (start_year + delta_year).to_pydatetime()

def get_legendre(nmax, mmax, theta, schmidtnormalize = True, negative_m = False, minlat = 0, keys = None):
    """ calculate associated Legendre functions 

        nmax             -- maximum total wavenumber
        mmax             -- maximum zonal wavenumber
        theta            -- colatitude in degrees, with N terms
        schmidtnormalize -- True if Schmidth seminormalization is wanted, False otherwise
        negative_m       -- True if you want the functions for negative m (complex expansion)
        keys             -- list of (n, m) tuples to return an array instead of a dict

        returns:
          P, dP -- dicts of legendre functions, and derivatives, with wavenumber tuple as keys
        
        or, if keys != None:
          PdP   -- array of size N, 2*M, where M is the number of terms. The first half of the 
                   columns are P, and the second half are dP


        algorithm from "Spacecraft Attitude Determination and Control" by James Richard Wertz
        
        could be unstable for large nmax...

        KML 2016-04-22

    """

    theta = theta.flatten()[:, np.newaxis]

    P = {}
    dP = {}
    sinth = np.sin(d2r*theta)
    costh = np.cos(d2r*theta)

    if schmidtnormalize:
        S = {}
        S[0, 0] = 1.

    # initialize the functions:
    for n in range(nmax +1):
        for m in range(nmax + 1):
            P[n, m] = np.zeros_like(theta, dtype = np.float64)
            dP[n, m] = np.zeros_like(theta, dtype = np.float64)

    P[0, 0] = np.ones_like(theta, dtype = np.float64)
    P[0, 0][np.abs(90 - theta) < minlat] = 0
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

            if schmidtnormalize:
                # compute Schmidt normalization
                if m == 0:
                    S[n, 0] = S[n - 1, 0] * (2.*n - 1)/n
                else:
                    S[n, m] = S[n, m - 1] * np.sqrt((n - m + 1)*(int(m == 1) + 1.)/(n + m))


    if schmidtnormalize:
        # now apply Schmidt normalization
        for n in range(1, nmax + 1):
            for m in range(0, min([n + 1, mmax + 1])):
                P[n, m]  *= S[n, m]
                dP[n, m] *= S[n, m]

    if negative_m:
        for n  in range(1, nmax + 1):
            for m in range(0, min([n + 1, mmax + 1])):
                P[n, -m]  = -1.**(-m) * factorial(n-m)/factorial(n+m) *  P[n, m]
                dP[n, -m] = -1.**(-m) * factorial(n-m)/factorial(n+m) * dP[n, m]


    if keys is None:
        return P, dP
    else:
        Pmat  = np.hstack(tuple(P[key] for key in keys))
        dPmat = np.hstack(tuple(dP[key] for key in keys)) 
    
        return np.hstack((Pmat, dPmat))



def read_shc(filename, return_all = False):
    """ read shc (spherical harmonic coefficient) file

        The function returns two pandas DataFrames, g, and h, containing the gauss coefficients.
        The column names are (n, m)
        The index is time (python datetimes).

        SECULAR VARIATION

        if return_all is True, the following will also be returned:
            N_MIN, N_MAX, NTIMES, SP_ORDER, N_STEPS

        The main point of using this format is that it will be easier to interpolate to other times.
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

    g = {key:coeffdict[key] for key in coeffdict.keys() if key[1] >= 0}               # g coefficients
    h = {(key[0], -key[1]):coeffdict[key] for key in coeffdict.keys() if key[1] < 0 } # h coefficients
    for key in [k for k in g.keys() if k[1] == 0]: # add zero coefficients for m = 0 in h dictionary
        h[key] = 0

    # g and h should now have the same number of keys, so this check should never fail:
    assert len(g.keys()) == len(h.keys())

    gdf = pd.DataFrame(g, index = times)
    hdf = pd.DataFrame(h, index = times)

    # make sure that the column keys of h are in same order as in g:
    hdf = hdf[gdf.columns]

    return gdf, hdf



def geod2geoc(gdlat, height, Bn, Bu):
    """
    theta, r, B_th, B_r = geod2lat(gdlat, height, X, Z)

       INPUTS:    
       gdlat is geodetic latitude (not colat)
       height is geodetic height (km)
       Bn is northward vector component in geodetic coordinates 
       Bu is upward vector component in geodetic coordinates

       OUTPUTS:
       theta is geocentric colatitude (degrees)
       r is geocentric radius (km)
       B_th is geocentric southward component (theta direction)
       B_r is geocentric radial component


    after Matlab code by Nils Olsen, DTU
    """

    a = WGS84_a
    b = a*np.sqrt(1 - WGS84_e2)

    sin_alpha_2 = np.sin(gdlat*d2r)**2
    cos_alpha_2 = np.cos(gdlat*d2r)**2

    # calculate geocentric latitude and radius
    tmp = height * np.sqrt(a**2 * cos_alpha_2 + b**2 * sin_alpha_2)
    beta = np.arctan((tmp + b**2)/(tmp + a**2) * np.tan(gdlat * d2r))
    theta = np.pi/2 - beta
    r = np.sqrt(height**2 + 2 * tmp + a**2 * (1 - (1 - (b/a)**4) * sin_alpha_2) / (1 - (1 - (b/a)**2) * sin_alpha_2))

    # calculate geocentric components
    psi  =  np.sin(gdlat*d2r) * np.sin(theta) - np.cos(gdlat*d2r) * np.cos(theta)
    
    B_r  = -np.sin(psi) * Bn + np.cos(psi) * Bu
    B_th = -np.cos(psi) * Bn - np.sin(psi) * Bu

    theta = theta/d2r

    return theta, r, B_th, B_r
 
def geoc2geod(theta, r, B_th, B_r):
    """
    gdlat, height, Bn, Bu = geod2lat(theta, r, B_th, B_r)

       INPUTS:    
       theta is geocentric colatitude (degrees)
       r is geocentric radius (km)
       B_r is geocentric radial component
       B_th is geocentric southward component (theta direction)

       OUTPUTS:
       gdlat is geodetic latitude (degrees, not colat)
       height is geodetic height (km)
       Bn is northward vector component in geodetic coordinates 
       Bu is upward vector component in geodetic coordinates


    after Matlab code by Nils Olsen, DTU
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
    SCL = np.sin(GCLAT * d2r)
    
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
    gdlat = DLTCL + GCLAT * d2r
    height = r * np.cos(DLTCL)- a * np.sqrt(1 -  E2 * np.sin(gdlat) ** 2)


    # magnetic components 
    psi = np.sin(gdlat) * np.sin(theta*d2r) - np.cos(gdlat) * np.cos(theta*d2r)
    Bn = -np.cos(psi) * B_th - np.sin(psi) * B_r 
    Bu = -np.sin(psi) * B_th + np.cos(psi) * B_r 

    gdlat = gdlat / d2r

    return gdlat, height, Bn, Bu



def igrf_gc(r, theta, phi, date, coeff_fn = shc_fn):
    """ compute magnetic field components at given dates, at given geocentric latitude and longitude:

        dates: python datetime(s). Can be an array (shape N, dim <= 1)
        glat:  geocentric latitude(s). Can be an array (shape M, dim <= 1)
        glon: geocentric longitude(s). Can be an array (shape M, dim <= 1)
        r: geocentric radius in km. Can be array (shape M, dim <= 1)
        coeffs: either tuple containing g and, h, or path to shc-file
        sp_order: order of the spline interpolation (DOESN'T WORK)

        r, theta, phi must be broadcastable
        r must be in km, theta and phi in degrees

        returns: X, Y, Z (north, east, down)
                 these are arrays with shape (N, M), where N is space and M is time

    """

    # read coefficient file:
    g, h = read_shc(coeff_fn)

    if np.any(date > g.index[-1]) or np.any(date < g.index[0]):
        print('Warning: You provided date(s) not covered by coefficient file \n({} to {})'.format(
              g.index[0].date(), g.index[-1].date()))

    if not hasattr(date, '__iter__'):
        date = [date] 

    # convert input to arrays in case they aren't
    r, theta, phi = tuple(map(lambda x: np.array(x, ndmin = 1), [r, theta, phi]))

    # get coordinate arrays to same size and shape
    shape = np.broadcast_shapes(r.shape, theta.shape, phi.shape)
    r, theta, phi = map(lambda x: np.broadcast_to(x, shape)   , [r, theta, phi])
    r, theta, phi = map(lambda x: x.flatten().reshape((-1 ,1)), [r, theta, phi]) # column vectors

    # make row vectors of wave numbers n and m:
    n, m = np.array([k for k in g.columns]).T
    n, m = n.reshape((1, -1)), m.reshape((1, -1))

    # get maximum N and maximum M:
    N, M = np.max(n), np.max(m)

    # get the legendre functions
    PdP = get_legendre(N, M, theta, schmidtnormalize = True, keys = g.keys())
    P, dP = np.split(PdP, 2, axis = 1)

    # Append coefficients at desired times (skip if index is already in coefficient data frame):
    index = g.index.union(date)

    g = g.reindex(index).groupby(index).first() # reindex and skip duplicates
    h = h.reindex(index).groupby(index).first() # reindex and skip duplicates

    # interpolate and collect the coefficients at desired times:
    g = g.interpolate(method = 'time').loc[date, :]
    h = h.interpolate(method = 'time').loc[date, :]

    # compute cosmlon and sinmlon:
    cosmphi = np.cos(phi * d2r * m) # shape (n_coords x n_model_params/2)
    sinmphi = np.sin(phi * d2r * m)

    # make versions of n and m that are repeated twice
    nn, mm = np.tile(n, 2), np.tile(m, 2)

    # calculate Br:
    G  = (REFRE / r) ** (nn + 2) * (nn + 1) * np.hstack((P * cosmphi, P * sinmphi))
    Br = G.dot(np.hstack((g.values, h.values)).T).T # shape (n_times, n_coords)

    # calculate Btheta:
    G  = -(REFRE / r) ** (nn + 1) * np.hstack((dP * cosmphi, dP * sinmphi)) \
         * REFRE / r
    Btheta = G.dot(np.hstack((g.values, h.values)).T).T # shape (n_times, n_coords)

    # calculate Bphi:
    G  = -(REFRE / r) ** (nn + 1) * mm * np.hstack((-P * sinmphi, P * cosmphi)) \
         * REFRE / r / np.sin(theta * d2r)
    Bphi = G.dot(np.hstack((g.values, h.values)).T).T # shape (n_times, n_coords)

    # reshape and return
    outshape = tuple([Bphi.shape[0]] + list(shape))
    return Br.reshape(outshape), Btheta.reshape(outshape), Bphi.reshape(outshape)


def igrf(lon, lat, h, date):
    """
    """

    # convert input to arrays and cast to same shape:
    lon, lat, h = tuple(map(lambda x: np.array(x, ndmin = 1), [lon, lat, h]))
    shape = np.broadcast_shapes(lon.shape, lat.shape, h.shape)
    lon, lat, h = map(lambda x: np.broadcast_to(x, shape), [lon, lat, h])
    lon, lat, h = map(lambda x: x.flatten(), [lon, lat, h])

    # convert to geocentric:
    theta, r, _, __ = geod2geoc(lat, h, h, h)
    phi = lon

    print(r, 90 - theta - lat)

    # calculate geocentric components of IGRF:
    Br, Btheta, Bphi = igrf_gc(r, theta, phi, date)
    Be = Bphi

    # convert output to geodetic
    lat_, h_, Bn, Bu = geoc2geod(theta, r, Btheta, Br)

    # return with shapes implied by input
    outshape = tuple([Be.shape[0]] + list(shape))
    return Be.reshape(outshape), Bn.reshape(outshape), Bu.reshape(outshape)



if __name__ == '__main__':
    
    from datetime import datetime

    dates = pd.date_range('1999-01-01', '2035-01-01', freq = '1Y') + pd.DateOffset(days = 1)
    lat = np.array([0, 45, 80])
    lon = 0
    h = 0

    Be, Bn, Bu = igrf(lon, lat, h, dates)

    k = pd.DataFrame({'Be':Be[:, 1], 'Bn':Bn[:, 1], 'Bu':Bu[:, 1]}, index = dates)
    print(k)


    print(igrf(30.3, 78.3, 131, datetime(2021, 3, 28)))
