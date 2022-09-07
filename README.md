# nd_line
interpolate points on an n-dimensional line by euclidean arc length

### Usage
from nd_line import nd_line

ln = nd_line(points) (where points is an array like with rows = points and columns = dimensions. An array with three points in 4 dimensions would have 3 rows and 4 colums)

methods:

- ln.interp(dist) returns a point dist length along the arc of the line

- ln.interp_rat(ratio) ratio should be a value between 0 and 1, returns a value ratio*length along the line

- ln.splineify(samples) generates a new line from a spline approximation, occurs in place, use samples to specify how many points will be sampled from the splines to generate the new line

attributes:

- ln.points: the points of the line
- ln.length: the length of the line
- ln.type: linear if not spline approximated, spline otherwise


### Notes

Currently points must be sampled one at a time, future version will allow interpolation of a list of distances along the line
