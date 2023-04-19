"""
Test function to compute inclination and declination.
"""
import pytest
import numpy as np
import numpy.testing as npt

from ppigrf import get_inclination_declination


@pytest.mark.parametrize("degrees", (True, False))
@pytest.mark.parametrize(
    "geomagnetic_field, inclination, declination",
    [
        [(0, 0, -1), 90, 0],
        [(0, 0, 1), -90, 0],
        [(1, 1, 0), 0, 45],
        [(-1, 1, 0), 0, -45],
        [(1, 1, -np.sqrt(2)), 45, 45],
        [(1, 1, np.sqrt(2)), -45, 45],
        [(-2_938.0, 16_308.1, -52_741.9), 72.5582, -10.2126],
    ],
)
def test_inclination_declination(degrees, geomagnetic_field, inclination, declination):
    """
    Test inclination and declination with known values.

    The last values were obtained by one of the online calculators.
    """
    if not degrees:
        inclination = np.radians(inclination)
        declination = np.radians(declination)
    inc, dec = get_inclination_declination(*geomagnetic_field, degrees=degrees)
    npt.assert_allclose([inc, dec], [inclination, declination], rtol=5e-5)
