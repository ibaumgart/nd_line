# nd_line
interpolate points on an n-dimensional line by euclidean arc length

### Usage
from nd_line import nd_line

ln = nd_line(points) (where points is an array like with rows = points and columns = dimensions. An array with three points in 4 dimensions would have 3 rows and 4 colums)

methods:

-ln.interp(dist) returns a point dist length along the arc of the line

-ln.splineify(samples) generates a new line from a spline approximation, occurs in place, use samples to specify how many points will be sampled from the splines to generate the new line

attributes:

-ln.points: the points of the line
-ln.length: the length of the line
-ln.type: linear if not spline approximated, spline otherwise


### Notes

Currently the interp method is not particularly efficient, and requires each point to be sampled individually, in the future I plan to address both of these limitations