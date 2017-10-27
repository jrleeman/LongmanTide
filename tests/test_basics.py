# Copyright (c) 2017 Leeman Geophysical LLC.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
r"""Basic functionality tests of the Longman tide model."""

from datetime import datetime
import numpy as np

from longmantide import longmantide


def test_basic():
    lat = 40.7914  # Station Latitude
    lon = 282.1414  # Station Longitude
    alt = 370.  # Station Altitude [meters]
    model = longmantide.TideModel()  # Make a model object
    time = datetime(2015, 4, 23, 0, 0, 0)  # When we want the tide
    gm, gs, g = model.solve_longman(lat, lon, alt, time)
    np.testing.assert_almost_equal(gm, 0.0324029651226, 8)
    np.testing.assert_almost_equal(gs, -0.0288682178454, 8)
    np.testing.assert_almost_equal(g, 0.00353474727722, 8)
