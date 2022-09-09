"""Tests for the nd_line module."""

import random

import numpy as np

from nd_line.nd_line import nd_line


class Test2D:
    """Test the nd_line class for a two-dimensional line."""

    def setup_method(self):
        """Set up the 2D line."""
        random.seed(a=123)
        pts = np.array([[random.random() for _ in range(100)] for _ in range(2)])
        pts = np.transpose(pts)
        self.line = nd_line(pts)

    def test_zero(self):
        """Test interpolation at zero distance."""
        np.testing.assert_allclose(self.line.interp(0), np.array([0.0523636, 0.75081494]))

    def test_length(self):
        """Test the line length calculation."""
        assert self.line.length == 51.784153048659896

    def test_interp(self):
        """Test interpolation at half distance."""
        np.testing.assert_allclose(self.line.interp(self.line.length / 2), np.array([0.11157182, 0.28764942]))

    def test_end(self):
        """Test interpolation at end distance."""
        np.testing.assert_allclose(self.line.interp(self.line.length), np.array([0.47251074, 0.41472736]))


class Test10D:
    """Test the nd_line class for a ten-dimensional line."""

    def setup_method(self):
        """Set up the 10D line."""
        random.seed(a=123)
        pts = np.array([[random.random() for _ in range(100)] for _ in range(10)])
        pts = np.transpose(pts)
        self.line = nd_line(pts)

    def test_zero(self):
        """Test interpolation at zero distance."""
        np.testing.assert_allclose(
            self.line.interp(0),
            np.array(
                [
                    0.0523636,
                    0.75081494,
                    0.5004748,
                    0.67094985,
                    0.20534254,
                    0.15618528,
                    0.15576653,
                    0.19688572,
                    0.66846312,
                    0.99964834,
                ]
            ),
        )

    def test_length(self):
        """Test the line length calculation."""
        assert self.line.length == 126.77686142601377

    def test_interp(self):
        """Test interpolation at half distance."""
        np.testing.assert_allclose(
            self.line.interp(self.line.length / 2),
            np.array(
                [
                    0.66056648,
                    0.45677264,
                    0.58577795,
                    0.20230632,
                    0.0345588,
                    0.61912402,
                    0.59964352,
                    0.1829004,
                    0.26046928,
                    0.68669252,
                ]
            ),
        )

    def test_end(self):
        """Test interpolation at end distance."""
        np.testing.assert_allclose(
            self.line.interp(self.line.length),
            np.array(
                [
                    0.47251074,
                    0.41472736,
                    0.11271949,
                    0.07060848,
                    0.67520735,
                    0.00524097,
                    0.77656087,
                    0.6270458,
                    0.76898746,
                    0.92127103,
                ]
            ),
            rtol=1e-06,
        )
