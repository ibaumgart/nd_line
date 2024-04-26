"""Module for creating an n-dimensional line.

Copyright Daniel Marshall
"""
from __future__ import annotations

import math
import typing as t

import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike
from scipy.interpolate import splev, splprep


class nd_line:
    """Class for n-dimensional line."""

    def __init__(self, points: ArrayLike, name: t.Optional[str] = None) -> None:
        """Create a line from a list of points.

        :param points: list of points
        :param name: line name
        """
        self.type: str = 'linear'  # legacy
        self.name = name  # editable
        # protected properties cannot be directly edited
        self._points: ndarray = np.array([tuple(x) for x in points])
        self._lengths: ndarray = self.compute_lengths(self.points)
        self._cumul: ndarray = np.concatenate(([0.0], np.cumsum(self._lengths)))
        self._length: float = float(self._cumul[-1])

    @property
    def points(self) -> ndarray:
        """Input points from which the line was constructed."""
        return self._points

    @property
    def lengths(self) -> ndarray:
        """Euclidean distance between each point along the line.

        Same length as nd_line.points.
        """
        return self._lengths

    @property
    def length(self) -> float:
        """Sum Euclidean length between each point along the line.

        Identical to nd_line.cumul[-1].
        """
        return self._length

    @property
    def cumul(self) -> ndarray:
        """Cumulative Euclidean length between each point along the line.

        Same length as nd_line.points.
        """
        return self._cumul

    @staticmethod
    def compute_lengths(points: ndarray) -> ndarray:
        """Calculate the length (sum of the Euclidean distance between points).

        :param points: numpy array of points
        :return: lengths between each line point.
        """
        lengths = [nd_line.e_dist(points[i], points[i + 1]) for i in range(len(points) - 1)]
        return np.array(lengths)

    def dist_from(self, point: ArrayLike) -> ndarray:
        """Calculate the distance between a given point and the points in the nd_line.

        :param point: numpy array of point
        :return: length of the line
        """
        translated_points = self._points - point
        return np.linalg.norm(translated_points, axis=1)

    def closest_idx(self, point: ndarray) -> int:
        """Get the index of the closest point in nd_line to a point.

        :param point: numpy array of point
        :return: index of closest point
        """
        return np.argmin(self.dist_from(point))

    def within_idx(self, point: ndarray, distance: float) -> ndarray:
        """Get the index of all points in nd_line within distance from point.

        :param point: numpy array of point
        :param distance: radius within which points will be returned
        :return: indices of nd_line points
        """
        dists = self.dist_from(point)
        return np.flatnonzero(dists[dists < distance])

    def contig_within_idx(self, point: ndarray, distance: float) -> ndarray:
        """Get the index of contiguous points in nd_line within distance from point around the closest point on nd_line.

        :param point: numpy array of point
        :param distance: radius within which points will be returned
        :return: indices of nd_line points
        """
        dists = self.dist_from(point)
        closest_idx = np.argmin(dists)
        start_idx = closest_idx
        while dists[start_idx] < distance and start_idx > 0:
            start_idx -= 1
        end_idx = closest_idx
        while dists[end_idx] < distance and end_idx < self._points.shape[0] - 1:
            end_idx += 1
        within_idx = np.arange(start_idx, end_idx + 1)
        return within_idx

    def interp(self, dist: float) -> ndarray:
        """Return a point a specified distance along the line.

        :param dist: distance along the line
        :type dist: float
        :return: numpy array of the point coordinates
        """
        assert dist <= self.length, 'length cannot be greater than line length'
        assert dist >= 0, 'length cannot be less than zero'
        if dist == 0:
            return self.points[0]
        if dist == self.length:
            return self.points[-1]
        index = np.where(self.cumul < dist)[0][-1]
        d = self.cumul[index]
        vector = (self.points[index + 1] - self.points[index]) / self.e_dist(self.points[index], self.points[index + 1])
        remdist = dist - d
        final_point = remdist * vector + self.points[index]
        return final_point

    def interp_rat(self, ratio: float) -> ndarray:
        """Return a point a specified ratio along the line.

        :param ratio: ratio along the line
        :return: numpy array of the point coordinates
        """
        assert 0 <= ratio <= 1, "Ratio for interp_rat() must be a value from 0 to 1"
        return self.interp(ratio * self.length)

    def resample(self, new_lengths: ArrayLike) -> nd_line:
        """Resample the line from new lengths along the line.

        :param new_lengths: vector of explicit resample lengths
        :return: new nd_line from resampled points
        """
        new_lengths = np.array(new_lengths)
        assert new_lengths.ndim == 1, "new_lengths must be a 1-D vector of lengths"
        assert np.all(np.logical_and(0.0 <= new_lengths, new_lengths <= self._length)), (
            "All new_lengths must between " "0 and nd_line length"
        )
        new_points = np.vstack(
            [np.interp(new_lengths, self._cumul, self._points[:, n]) for n in range(self._points.shape[1])]
        ).T
        return nd_line(new_points, name=self.name)

    def splineify(self, samples: t.Optional[int] = None, s: t.Optional[float] = 0, **kwargs) -> nd_spline:
        """Turn line into a spline approximation, returns new object.

        :param samples: number of samples to use for spline approximation
        :param s: smoothing factor for spline approximation
        """
        nds = nd_spline(self._points, name=self.name, s=s, **kwargs)
        if samples is not None:
            nds = nds.resample(samples)

        return nds

    @staticmethod
    def e_dist(a: ndarray, b: ndarray) -> float:
        """Calculate the euclidean distance between two points.

        :param a: numpy array of point a
        :param b: numpy array of point b
        :return: euclidean distance between a and b
        """
        return math.sqrt(sum((a - b) ** 2))


class nd_spline(nd_line):
    """Class for n-dimensional spline."""

    def __init__(self, points: ArrayLike, name: t.Optional[str] = None, s: t.Optional[float] = 0.0, **kwargs) -> None:
        """Create a spline from a list of points.

        :param points: list of points :param name: line name :param s: smoothing factor for spline :param **kwargs:
        keyword arguments for splprep (https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep
        .html). kwargs u will be overriden by np.linspace(0,1,len(points)).
        """
        super().__init__(points, name=name)
        self.type = 'spline'
        # protected attributes
        self._u = self._cumul / self._length
        self._s = s

        #
        if 's' in kwargs.keys():
            del kwargs['s']
        if 'u' in kwargs.keys():
            del kwargs['u']

        tck, _ = splprep(self._points.T, u=self._u, s=s, **kwargs)
        self._tck = tck

    @property
    def tck(self) -> tuple:
        """(t,c,k) tuple from splprep (https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.

        .splprep.html)
        """
        return self._tck

    @property
    def u(self) -> ndarray:
        """ratio from 0 to 1 at each point parameterizing the nd_spline (
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep.html)"""
        return self._u

    @property
    def s(self) -> float:
        """smoothing parameter from spline generation (
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep.html)"""
        return self._s

    def resample(self, new_lengths: ArrayLike) -> nd_spline:
        """Resample the spline from new lengths along the line.

        :param new_lengths: vector of explicit resample lengths
        :return: new nd_spline from resampled points
        """
        new_lengths = np.array(new_lengths)
        assert new_lengths.ndim == 1, "new_lengths must be a 1-D vector of lengths"
        assert np.all(np.logical_and(0.0 <= new_lengths, new_lengths <= self._length)), (
            "All new_lengths must between " "0 and nd_line length"
        )
        new_u = new_lengths / self._length
        new_points = np.array(splev(new_u, self._tck)).T
        return nd_spline(new_points, name=self.name, s=0)

    def recursive_upsample(self, tol: t.Optional[float] = 0.01) -> nd_spline:
        """Add a point between existing points until the curve is reasonably represented by nd_line.points.
        This will ensure that nd_line.lengths are representative of the length along the curve.

        :param tol: incremental relative tolerance at which to stop incrementing upsample.
        :return: new nd_spline from resampled points
        """

        def recursive_bisection(u1, u3, pt1, pt3, d13, tck, r_tol):
            u2 = (u1 + u3) / 2
            pt2 = np.array(splev(u2, tck)).T
            d12 = self.e_dist(pt1, pt2)
            d23 = self.e_dist(pt2, pt3)
            err = (d12 + d23 - d13) / d13
            if err > r_tol:
                return recursive_bisection(u1, u2, pt1, pt2, d12, tck, r_tol) + recursive_bisection(
                    u2, u3, pt2, pt3, d23, tck, r_tol
                )
            else:
                return [u3]

        new_u = [self._u[0]]
        for i in range(0, self._u.shape[0] - 1):
            new_u.extend(
                recursive_bisection(
                    self._u[i], self._u[i + 1], self._points[i], self._points[i + 1], self._lengths[i], self._tck, tol
                )
            )

        return self.resample(np.array(new_u) * self._length)

    def interp(self, dist: float) -> ndarray:
        """Return a point a specified distance along the line.

        :param dist: distance along the line
        :type dist: float
        :return: numpy array of the point coordinates
        """
        assert dist <= self._length, 'length cannot be greater than line length'
        assert dist >= 0, 'length cannot be less than zero'
        if dist == 0:
            return self._points[0]
        if dist == self._length:
            return self._points[-1]
        u = dist / self._length
        return self.interp_rat(u)

    def interp_rat(self, ratio: float) -> ndarray:
        """Return a point a specified ratio along the line.

        :param ratio: ratio along the line
        :return: numpy array of the point coordinates
        """
        assert 0 <= ratio <= 1, "Ratio for interp_rat() must be a value from 0 to 1"
        return np.array(splev(ratio, self._tck)).T

    def to_line(self) -> nd_line:
        """Return a nd_line representation of the nd_spline.

        :return: nd_line of nd_spline
        """
        return nd_line(self.points, name=self.name)
