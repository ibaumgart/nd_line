"""Tests for the nd_line module."""

import inspect
import random
import sys

import numpy as np

sys.path.append('.')

from src.nd_line.nd_line import nd_line, nd_spline  # noqa E402


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

    def test_dist_from(self):
        """Test the closest point function calculation."""
        halfway = self.line.points.shape[0] // 2
        close_pt = self.line.points[halfway].copy()
        dists = self.line.dist_from(close_pt)
        assert dists[halfway] == 0.0

    def test_closest(self):
        """Test the closest point function calculation."""
        halfway = self.line.points.shape[0] // 2
        close_pt = self.line.points[halfway]
        closest_idx = self.line.closest_idx(close_pt)
        assert np.array(self.line.points[closest_idx] == close_pt).all()
        assert closest_idx == halfway

    def test_contig_within(self):
        halfway = self.line.points.shape[0] // 2
        close_pt = self.line.points[halfway]
        contig_idx = self.line.contig_within_idx(close_pt, 0.1)
        assert halfway in contig_idx
        assert np.all(np.diff(contig_idx) == 1)

    def test_interp(self):
        """Test interpolation at half distance."""
        np.testing.assert_allclose(self.line.interp(self.line.length / 2), np.array([0.11157182, 0.28764942]))

    def test_end(self):
        """Test interpolation at end distance."""
        np.testing.assert_allclose(self.line.interp(self.line.length), np.array([0.47251074, 0.41472736]))

    def test_resample(self):
        u = np.arange(self.line.points.shape[0])
        new_u = np.arange(self.line.points.shape[0] - 0.5, step=0.5)
        new_lengths = np.interp(new_u, u, self.line.cumul)
        new_ndl = self.line.resample(new_lengths)
        assert np.all(new_ndl.cumul == new_lengths)
        assert np.all(new_ndl.cumul[::2] == self.line.cumul)


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

    def test_resample(self):
        u = np.arange(self.line.points.shape[0])
        new_u = np.arange(self.line.points.shape[0] - 0.5, step=0.5)
        new_lengths = np.interp(new_u, u, self.line.cumul)
        new_ndl = self.line.resample(new_lengths)
        assert np.all(new_ndl.cumul == new_lengths)
        assert np.all(new_ndl.cumul[::2] == self.line.cumul)

    def test_dist_from(self):
        """Test the closest point function calculation."""
        halfway = self.line.points.shape[0] // 2
        close_pt = self.line.points[halfway].copy()
        dists = self.line.dist_from(close_pt)
        assert dists[halfway] == 0.0

    def test_closest(self):
        """Test the closest point function calculation."""
        halfway = self.line.points.shape[0] // 2
        close_pt = self.line.points[halfway]
        closest_idx = self.line.closest_idx(close_pt)
        assert np.array(self.line.points[closest_idx] == close_pt).all()
        assert closest_idx == halfway

    def test_contig_within(self):
        halfway = self.line.points.shape[0] // 2
        close_pt = self.line.points[halfway]
        contig_idx = self.line.contig_within_idx(close_pt, 0.1)
        assert halfway in contig_idx
        assert np.all(np.diff(contig_idx) == 1)


class TestSpline2D:
    """Test the nd_spline class for a two-dimensional spline."""

    def setup_method(self):
        """Set up the 2D spline."""
        # Make a 3/4 circle
        uu = np.linspace(0, 3 * np.pi / 2, 10)
        pts = np.array([np.cos(uu) + 2, np.sin(uu) + 2])
        pts = np.transpose(pts)
        self.spline = nd_spline(pts)

    def test_resample(self):
        true_len = np.linspace(0, 3 * np.pi / 2, 100)
        new_len = np.linspace(0, self.spline.length, 100)
        resamp_spl = self.spline.resample(new_len)
        truth = np.array([np.cos(true_len) + 2, np.sin(true_len) + 2]).T
        np.testing.assert_allclose(resamp_spl.points, truth, rtol=1e-3, atol=1e-3)

    def test_recursive_upsample(self):
        tol = 1e-4
        true_len = 3 * np.pi / 2
        init_err = true_len - self.spline.length
        up_spline = self.spline.recursive_upsample(tol=tol)
        import matplotlib.pyplot as plt
        plt.plot()
        fin_err = true_len - up_spline.length
        err_pct = (init_err - fin_err) / init_err
        assert init_err * tol < err_pct
        print(f"recursive_upsample\n\tinitial error: {init_err}\n\tfinal error: {fin_err}")

    def test_interp(self):
        up_spline = self.spline.recursive_upsample()
        np.testing.assert_allclose(up_spline.interp(np.pi / 2), [2, 3], atol=0.001, rtol=0.001)

    def test_interp_rat(self):
        np.testing.assert_allclose(self.spline.interp_rat(1/3), [2, 3])


if __name__ == '__main__':
    t2d = Test2D()
    attrs = (getattr(t2d, name) for name in dir(t2d))
    methods = filter(inspect.ismethod, attrs)
    for method in methods:
        method()
        print(method)

    t10d = Test10D()
    attrs = (getattr(t10d, name) for name in dir(t10d))
    methods = filter(inspect.ismethod, attrs)
    for method in methods:
        method()
        print(method)

    ts2d = TestSpline2D()
    attrs = (getattr(ts2d, name) for name in dir(ts2d))
    methods = filter(inspect.ismethod, attrs)
    for method in methods:
        method()
        print(method)
