# nd_line

Interpolate points on an n-dimensional line by Euclidean arc length.

### Installation

`pip install nd_line`

#### Classes
- `nd_line`: Linear representation of ordered n-dimensional points.
- `nd_spline`: B-Spline representation of ordered n-dimensional points. Child class of nd_line.

#### Methods

##### Utilities
- `nd_line.dist_from(point)`: returns the distance from all ndl points to the given point.

- `nd_line.closest_idx(point)`: returns the index of the closest of nd_line.points to the given point.

- `nd_line.within_idx(point, dist)`: returns the indices of all nd_line.points within distance of the given point.

- `nd_line.contig_within_idx(point, dist)`: same as within_idx, but indices are consecutive and include the closest point.

##### Linear Interpolation
- `nd_line.interp(dist)`: returns a point(s) dist(s) length along the Euclidean arc of the line

- `nd_line.interp_rat(ratio)`: ratio(s) should be a value between 0 and 1, returns a value ratio*length along the line

- `nd_line.resample(dists)`: generates a new nd_line object with points interpolated from the original nd_line.

#### Spline Interpolation
- `nd_line.to_spline(samples)`: generates a new nd_spline from the nd_line points, use samples to specify how many points will be sampled from the splines to generate the new line.

- `nd_spline.recursive_upsample(tol)`: generates a new nd_spline by upsampling the spline until the Euclidean arc length asymptotes. use tol to specify the proportional length difference at which to stop upsampling.

- `nd_spline.interp_rat(ratio)`: same as nd_line.interp_rat except using the spline with scipy.interpolate.splev.

- `nd_spline.interp(dist)`: same as nd_line.interp except using the spline. _recommended to use nds.recursive_upsample before interpolating distances._

- `nd_spline.resample(dists)`: same as nd_line.resample except using the spline. _recommended to use nds.recursive_upsample before interpolating distances._

#### Attributes

##### nd_line and nd_spline
- `nd_line.points`: the points of the line
- `nd_line.length`: the length of the line computed as Euclidean distance between points
- `nd_line.lengths`: the Euclidean distance between each point of the line
- `nd_line.cumul`: the cumulative Euclidean distance from the first point to each point of the line
- `nd_line.type`: linear if not spline approximated, spline otherwise

##### nd_spline only
- `nd_spline.tck`: the knots, coefficients, and degree of the spline
- `nd_spline.u`: the underlying parameter of the spline from 0 (first point) to 1 (last point)

#### nd_line Example

```python
from nd_line.nd_line import nd_line
import numpy as np

ndl = nd_line(np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]))
interpolated_point = ndl.interp(1.5)
interpolated_points = ndl.interp([0.5, 1.5])
line_length = ndl.length
halfway_point = ndl.interp_rat(0.5)
```
#### nd_spline Example

```python
from nd_line.nd_spline import nd_spline
import numpy as np

# Make a semi-circle
uu = np.linspace(0, np.pi, 8)
# input points with shape (n_points, n_dimensions)
points = np.array([np.cos(uu), np.sin(uu)]).T

nds = nd_spline(points)
halfway_point = nds.interp_rat(1.5) # [0, 1]

# Length of a semicircle with radius=1 should be pi
nds.length # Not close to pi because points are sparse
upsampled_spline = nds.recursive_upsample() # upsample to estimate spline arc length with Euclidean arc length
upsampled_spline.length # Close to pi
```

### Notes

