"""
Test IGRF field
"""
import os
import pytest
import numpy as np
import numpy.testing as npt
import pandas as pd
from pathlib import Path
from datetime import datetime

from ppigrf import igrf
from ppigrf.ppigrf import yearfrac_to_datetime

# Define paths to test directory and test data directory
TEST_DIR = Path(os.path.dirname(__file__))
TEST_DATA_DIR = TEST_DIR / "data"


def load_precomputed_igrf(date):
    """
    Loads the precomputed IGRF files from a given date

    Available dates:
        * 2020-01-01
        * 2022-10-05

    Parameters
    ----------
    date : :class:`datetime.datetime` object
        Date of the precomputed IGRF field files that will be loaded.

    Returns
    -------
    igrf_precomputed : :class:`pandas.Dataframe`
        Dataframe containing the precomputed values of the IGRF on the given
        date.
    """
    # Read the csv files
    date_dir = TEST_DATA_DIR / date.strftime("%Y-%m-%d")
    first_columns = ["date", "latitude", "longitude", "altitude_km"]
    components = ("b_e", "b_n", "b_z")
    dataframes = []
    for component in components:
        columns = first_columns + [component, component + "_sv"]
        fname = date_dir / f"{component}.csv"
        df = pd.read_csv(fname, skiprows=13, names=columns)
        dataframes.append(df)
    # Merge the dataframes
    igrf_precomputed = pd.merge(dataframes[0], dataframes[1])
    igrf_precomputed = pd.merge(igrf_precomputed, dataframes[-1])
    # Convert the data in the dataframe into a datetime object
    decimal_date = igrf_precomputed.date.values[0]
    (date,) = yearfrac_to_datetime([decimal_date])
    igrf_precomputed = igrf_precomputed.assign(date=date)
    return igrf_precomputed


class TestIGRFKnownValues:
    """
    Test the IGRF field against precomputed values
    """

    @pytest.mark.parametrize(
        "date",
        [datetime(2020, 1, 1), datetime(2022, 10, 5)],
        ids=["2020-01-01", "2022-10-05"],
    )
    def test_igrf(self, date):
        """
        Test IGRF against the precomputed values

        The test on 2020-01-01 doesn't involve any interpolation on the
        dates.
        """
        # Get precomputed IGRF field
        precomputed_igrf = load_precomputed_igrf(date)
        # Overwrite the date with the one in the data file
        # date = precomputed_igrf.date.values[0]
        # Compute igrf using ppigrf
        b_e, b_n, b_u = igrf(
            precomputed_igrf.longitude,
            precomputed_igrf.latitude,
            precomputed_igrf.altitude_km,
            date,
        )
        # Ravel the arrays
        b_e, b_n, b_u = tuple(np.ravel(component) for component in (b_e, b_n, b_u))
        # Check if the values are equal to the expected ones
        rtol = 1e-2  # 1% of error
        atol = 1  # 1nT absolute error (for points where the component is close to 0)
        npt.assert_allclose(b_e, precomputed_igrf.b_e, rtol=rtol, atol=atol)
        npt.assert_allclose(b_n, precomputed_igrf.b_n, rtol=rtol, atol=atol)
        npt.assert_allclose(
            b_u, -precomputed_igrf.b_z, rtol=rtol, atol=atol
        )  # invert the direction of b_z
