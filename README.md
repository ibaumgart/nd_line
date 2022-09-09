# nd_line
Interpolate points on an n-dimensional line by euclidean arc length.

### Installation
`pip install nd_line`

### Usage

#### Methods:

- `ln.interp(dist)`: returns a point dist length along the arc of the line

- `ln.interp_rat(ratio)`: ratio should be a value between 0 and 1, returns a value ratio*length along the line

- `ln.splineify(samples)`: generates a new line from a spline approximation, occurs in place, use samples to specify how many points will be sampled from the splines to generate the new line

#### Attributes:

- `ln.points`: the points of the line
- `ln.length`: the length of the line
- `ln.type`: linear if not spline approximated, spline otherwise

#### Example:
```python
from nd_line.nd_line import nd_line
import numpy as np

ln = nd_line(np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]))
interpolated_point = ln.interp(1.5)
line_length = ln.length
halfway_point = ln.interp_rat(0.5)
```

### Notes

Currently points must be sampled one at a time, future version will allow interpolation of a list of distances along the line
