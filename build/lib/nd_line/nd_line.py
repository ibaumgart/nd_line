# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 14:07:13 2021

@author: dpm42
"""

import numpy as np
from scipy.spatial import distance
import sys
from scipy.interpolate import splev, splprep
import matplotlib.pyplot as plt

class nd_line():
    def __init__(self,points,inplace = False):
        self.points = np.array([tuple(x) for x in points])
        self.length = self._length(self.points)
        self.type = 'linear'
    def _length(self,points):
        'calculate the length (sum of the euclidean distance between points)'
        return  sum([self.e_dist(points[i],points[i+1]) for i in range(len(points)-1)])
    def interp(self,dist):
        'return a point a specified distance along the line'
        if dist>self._length(self.points): sys.exit('length cannot be greater than line length')
        if dist==0: return self.points[0]
        i=0
        d=0
        while d<dist:
            i+=1
            d+=self.e_dist(self.points[i-1],self.points[i])
        last_point_dist = self.e_dist(self.points[i-1],self.points[i])
        d-=last_point_dist
        vector = (self.points[i]-self.points[i-1])/last_point_dist
        remdist = dist-d
        final_point = remdist*vector+self.points[i-1]
        check = self.points.copy()[0:i+1]
        check[i] = final_point
        if abs(self._length(check)/dist-1)>.001: sys.exit('Something is wrong')
        return(final_point)
    def interp_rat(self,ratio):
        return self.interp(ratio*self.length)
    def splineify(self,samples = None,s=0):
        'Turn line into a spline approximation, currently occurs in place'
        if samples is None: samples = len(self.points)
        tck,u = splprep([self.points[:,i] for i in range(self.points.shape[1])],s=s)
        self.points = np.transpose(splev(np.linspace(0,1,num=samples),tck))
        self.length = self._length(self.points)
        self.type = 'spline'
    def plot2d(self):
        fig = plt.figure()
        plt.scatter(self.points[:,0],self.points[:,1])
    def plot3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(self.points[:,0],self.points[:,1],self.points[:,2])
    def e_dist(self,a,b):
        return np.sqrt(np.sum((a - b) ** 2, axis=0))