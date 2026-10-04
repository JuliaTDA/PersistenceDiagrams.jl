"""Generate independent, pinned GUDHI numerical reference fixtures."""
from pathlib import Path
import csv
import numpy as np
import gudhi
from gudhi.representations.metrics import (
    _sliced_wasserstein_distance_on_projections, _persistence_fisher_distance,
)

assert gudhi.__version__ == "3.11.0"
root = Path(__file__).resolve().parent
rng = np.random.default_rng(20261003)
points, cases = [], []
for case, (n, m) in enumerate([(0, 0), (0, 3), (1, 1), (3, 5), (20, 15), (30, 30)]):
    diagrams = []
    for side, count in enumerate((n, m)):
        b = rng.uniform(-1, 2, count)
        d = b + rng.uniform(0.01, 2, count)
        diagram = np.column_stack((b, d))
        diagrams.append(diagram)
        points.extend((case, side, x, y) for x, y in diagram)
    left, right = diagrams
    for slices in (2, 50, 200):
        # Use the complete documented uniform grid. GUDHI 3.11's high-level
        # direction builder drops its last direction; its projection-distance
        # primitive lets us compare exactly the same quadrature instead.
        angles = -np.pi/2 + np.arange(slices)*np.pi/slices
        lines = np.vstack((np.cos(angles), np.sin(angles)))
        projections = [np.vstack((d @ lines, (d @ (0.5*np.ones((2, 2)))) @ lines))
                       for d in diagrams]
        sw = _sliced_wasserstein_distance_on_projections(*projections)
        fisher = 1.0 if n+m == 0 else np.exp(-_persistence_fisher_distance(left, right, bandwidth=0.7)/1.2)
        cases.append((case, n, m, slices, sw, np.exp(-sw/(2*0.8**2)), fisher))
with (root/"gudhi_points.csv").open("w") as f:
    writer = csv.writer(f); writer.writerow(("case", "side", "birth", "death")); writer.writerows(points)
with (root/"gudhi_expected.csv").open("w") as f:
    writer = csv.writer(f)
    writer.writerow(("case", "n", "m", "slices", "sw", "sw_kernel", "fisher_kernel"))
    writer.writerows(cases)
print(f"GUDHI {gudhi.__version__}: {len(cases)} reference cases generated (seed 20261003)")
