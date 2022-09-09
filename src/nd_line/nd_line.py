"""Module for creating an n-dimensional line."""

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import splev, splprep


class nd_line:
    """Class for n-dimensional line."""

    def __init__(self, points):
        """Create a line from a list of points.

        :param points: list of points
        """
        self.points = np.array([tuple(x) for x in points])
        alldist = self._lengths(self.points)
        self.length = sum(alldist)
        self.cumul = np.cumsum([0] + alldist)
        self.type = 'linear'

    def _lengths(self, points):
        """Calculate the length (sum of the euclidean distance between points).

        :param points: numpy array of points
        :return: length of the line
        """
        return [self.e_dist(points[i], points[i + 1]) for i in range(len(points) - 1)]

    def _length(self, points):
        """Calculate the length (sum of the euclidean distance between points).

        :param points: numpy array of points
        :type points: ndarray
        :return: length of the line
        """
        return sum([self.e_dist(points[i], points[i + 1]) for i in range(len(points) - 1)])

    def interp(self, dist):
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

    def interp_rat(self, ratio):
        """Return a point a specified ratio along the line.

        :param ratio: ratio along the line
        :return: numpy array of the point coordinates
        """
        assert ratio <= 1, "Ratio for interp_rat() must be a value from 0 to 1"
        return self.interp(ratio * self.length)

    def splineify(self, samples=None, s=0):
        """Turn line into a spline approximation, currently occurs in place.

        :param samples: number of samples to use for spline approximation
        :param s: smoothing factor for spline approximation
        """
        if samples is None:
            samples = len(self.points)
        tck, u = splprep([self.points[:, i] for i in range(self.points.shape[1])], s=s)
        self.points = np.transpose(splev(np.linspace(0, 1, num=samples), tck))
        self.length = self._length(self.points)
        self.type = 'spline'

    def plot2d(self):
        """Plot the line in 2D."""
        assert self.points.shape[1] == 2, 'Line must be 2D to plot in 2D'
        plt.figure()
        plt.scatter(self.points[:, 0], self.points[:, 1])

    def plot3d(self):
        """Plot the line in 3D."""
        assert self.points.shape[1] == 3, 'Line must be 3D to plot in 3D'
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2])

    @staticmethod
    def e_dist(a, b):
        """Calculate the euclidean distance between two points.

        :param a: numpy array of point a
        :param b: numpy array of point b
        :return: euclidean distance between a and b
        """
        return np.sqrt(np.sum((a - b) ** 2, axis=0))
