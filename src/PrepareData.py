import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from matplotlib.path import Path as PolygonPath


def read_exp_boundary(exp_path):
    """
    Read one ordered polygon from an ISSM/Argus .exp file.

    Returns
    -------
    boundary_xy : ndarray, shape (N, 2)
        Ordered polygon coordinates, explicitly closed.
    """
    with open(exp_path, "r") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]

    if not lines:
        raise ValueError(f"No boundary data found in {exp_path}")

    # First numeric line: number of points and profile value.
    header = lines[0].split()
    n_points = int(header[0])

    if n_points < 3:
        raise ValueError("Boundary must contain at least three points.")

    # This reader expects exactly one polygon/profile.
    if len(lines) - 1 != n_points:
        raise ValueError(
            f"Expected {n_points} coordinate rows, found {len(lines) - 1}. "
            "This reader supports a single polygon/profile."
        )

    boundary = np.asarray(
        [list(map(float, line.split())) for line in lines[1:]],
        dtype=float,
    )

    if boundary.shape != (n_points, 2):
        raise ValueError("Each boundary row must contain exactly x and y.")
    if not np.all(np.isfinite(boundary)):
        raise ValueError("Boundary coordinates must be finite.")

    # Your example is already closed; avoid adding a duplicate endpoint.
    if not np.array_equal(boundary[0], boundary[-1]):
        boundary = np.vstack((boundary, boundary[0]))

    return boundary


def pixelate_simulation(
    sim,
    nx=256,
    ny=256,
    exp_path=None,
    boundary_xy=None,
    extent=None,
    fields=None,
    x_key="x_vertex",
    y_key="y_vertex",
):
    """
    Interpolate vertex fields onto an (ny, nx) pixel-center grid.

    Values outside the supplied polygon are set to zero.

    Parameters
    ----------
    sim : mapping
        Mesh coordinates and numeric vertex fields.
    nx, ny : int
        Number of pixels along x and y.
    exp_path : str or Path, optional
        Path to a single-profile .exp boundary file.
        Supply exactly one of exp_path or boundary_xy.
    boundary_xy : array-like, shape (N, 2), optional
        Ordered boundary coordinates, optionally already closed.
    extent : tuple, optional
        (xmin, xmax, ymin, ymax).
        Defaults to the boundary's bounding box.
    fields : str or sequence of str, optional
        Fields to interpolate. If None, automatically select real numeric
        vectors with one entry per mesh vertex, excluding coordinate keys.
        MATLAB-style row and column vectors are supported.
    x_key, y_key : str
        Keys containing mesh vertex coordinates.

    Returns
    -------
    dict
        Each selected field : array with shape (ny, nx).
        mask, X, Y          : arrays with shape (ny, nx).
        x, y                : pixel-center coordinates.
        x_edges, y_edges    : pixel-edge coordinates.
        boundary_xy         : closed polygon coordinates.
        extent              : grid bounds.

    Notes
    -----
    Pixel centers inside the polygon but outside the mesh convex hull
    receive NaN. Nonfinite source field values can also propagate.
    """
    for name, n in (("nx", nx), ("ny", ny)):
        if (
            isinstance(n, (bool, np.bool_))
            or not isinstance(n, (int, np.integer))
            or n <= 0
        ):
            raise ValueError(f"{name} must be a positive integer.")

    if (exp_path is None) == (boundary_xy is None):
        raise ValueError("Supply exactly one of exp_path or boundary_xy.")

    if exp_path is not None:
        boundary = np.asarray(read_exp_boundary(exp_path), dtype=float)
    else:
        boundary = np.asarray(boundary_xy, dtype=float)

    if (
        boundary.ndim != 2
        or boundary.shape[1] != 2
        or boundary.shape[0] < 3
        or not np.isfinite(boundary).all()
    ):
        raise ValueError("Boundary must be a finite (N, 2) array.")

    if not np.array_equal(boundary[0], boundary[-1]):
        boundary = np.vstack((boundary, boundary[0]))

    # Flatten MATLAB-style row/column coordinate vectors.
    x = np.asarray(sim[x_key], dtype=float).ravel()
    y = np.asarray(sim[y_key], dtype=float).ravel()

    if x.size != y.size:
        raise ValueError("x and y coordinates must have the same length.")

    if x.size < 3 or not (
        np.isfinite(x).all() and np.isfinite(y).all()
    ):
        raise ValueError("At least three finite mesh vertices are required.")

    n_vertices = x.size

    def is_vertex_field(value):
        """Recognize real numeric vectors, including row/column vectors."""
        try:
            arr = np.asarray(value)
        except (TypeError, ValueError):
            return False

        is_vector = (
            arr.ndim == 1
            or (arr.ndim == 2 and 1 in arr.shape)
        )
        is_real_numeric = (
            np.issubdtype(arr.dtype, np.number)
            and not np.iscomplexobj(arr)
        )

        return (
            is_vector
            and is_real_numeric
            and arr.size == n_vertices
        )

    if fields is None:
        field_names = [
            name
            for name, value in sim.items()
            if name not in (x_key, y_key) and is_vertex_field(value)
        ]
    elif isinstance(fields, str):
        field_names = [fields]
    else:
        field_names = list(fields)

    if not field_names:
        raise ValueError("No vertex fields found or selected.")

    if len(set(field_names)) != len(field_names):
        raise ValueError("Field names must be unique.")

    # Avoid silently overwriting grid metadata in the returned dictionary.
    reserved_keys = {
        "x", "y", "X", "Y", "x_edges", "y_edges",
        "mask", "boundary_xy", "extent",
    }

    conflicts = reserved_keys.intersection(field_names)
    if conflicts:
        raise ValueError(
            "Field names conflict with output grid keys: "
            f"{sorted(conflicts)}. Rename these fields before interpolation."
        )

    values = []
    for name in field_names:
        if name in (x_key, y_key):
            raise ValueError(f"{name!r} is a coordinate key, not a field.")

        if name not in sim:
            raise KeyError(f"Field {name!r} is missing from sim.")

        if not is_vertex_field(sim[name]):
            raise ValueError(
                f"Field {name!r} must be a real numeric vector "
                f"with {n_vertices} entries."
            )

        values.append(np.asarray(sim[name], dtype=float).ravel())

    # Shape: (n_vertices, n_fields).
    values = np.column_stack(values)

    if extent is None:
        extent = (
            boundary[:, 0].min(),
            boundary[:, 0].max(),
            boundary[:, 1].min(),
            boundary[:, 1].max(),
        )

    xmin, xmax, ymin, ymax = map(float, extent)

    if not (
        np.isfinite([xmin, xmax, ymin, ymax]).all()
        and xmin < xmax
        and ymin < ymax
    ):
        raise ValueError("extent must have finite, increasing bounds.")

    x_edges = np.linspace(xmin, xmax, nx + 1)
    y_edges = np.linspace(ymin, ymax, ny + 1)

    x_grid = (x_edges[:-1] + x_edges[1:]) / 2
    y_grid = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(x_grid, y_grid)

    polygon = PolygonPath(boundary, closed=True)
    mask = polygon.contains_points(
        np.column_stack((X.ravel(), Y.ravel()))
    ).reshape(ny, nx)

    # Reuse one triangulation for all selected fields.
    triangulation = Delaunay(np.column_stack((x, y)))
    interpolator = LinearNDInterpolator(
        triangulation,
        values,
        fill_value=np.nan,
    )

    pixel_fields = np.zeros((ny, nx, len(field_names)), dtype=float)

    if mask.any():
        pixel_fields[mask] = interpolator(X[mask], Y[mask])

    result = {
        "x": x_grid,
        "y": y_grid,
        "X": X,
        "Y": Y,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "mask": mask,
        "boundary_xy": boundary,
        "extent": (xmin, xmax, ymin, ymax),
    }

    result.update({
        name: pixel_fields[..., j]
        for j, name in enumerate(field_names)
    })

    return result



def pixelate_evidence(grid, thawed_x, thawed_y, frozen_x, frozen_y):
    """
    Assign thawed/frozen evidence points to existing pixels.

    Repeated evidence of the same type in a pixel has no additional effect.
    Points outside the grid or in pixels outside the domain mask are ignored.

    Returns
    -------
    dict
        thawed_mask : bool array, shape (ny, nx)
        frozen_mask : bool array, shape (ny, nx)
        evidence_mask : bool array, True where either type occurs
        conflict_mask : bool array, True where both types occur
        labels : int8 array
            -1 = no evidence (including outside-domain pixels)
             0 = frozen only
             1 = thawed only
             2 = both thawed and frozen
    """
    x_edges = np.asarray(grid["x_edges"], dtype=float)
    y_edges = np.asarray(grid["y_edges"], dtype=float)
    domain_mask = np.asarray(grid["mask"], dtype=bool)

    ny, nx = domain_mask.shape

    if x_edges.size != nx + 1 or y_edges.size != ny + 1:
        raise ValueError("Grid edges do not match the domain mask shape.")

    def rasterize(x, y):
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()

        if x.size != y.size:
            raise ValueError("Evidence x and y must have the same length.")

        # Filter before indexing, including nonfinite coordinates.
        valid = (
            np.isfinite(x)
            & np.isfinite(y)
            & (x >= x_edges[0])
            & (x <= x_edges[-1])
            & (y >= y_edges[0])
            & (y <= y_edges[-1])
        )
        x, y = x[valid], y[valid]

        # Array indexing is [row_y, column_x].
        cols = np.searchsorted(x_edges, x, side="right") - 1
        rows = np.searchsorted(y_edges, y, side="right") - 1

        # Include points exactly on the grid's upper/right outer edges.
        cols = np.clip(cols, 0, nx - 1)
        rows = np.clip(rows, 0, ny - 1)

        evidence = np.zeros((ny, nx), dtype=bool)
        evidence[rows, cols] = True

        # Retain evidence only in pixels inside the domain.
        evidence &= domain_mask
        return evidence

    thawed_mask = rasterize(thawed_x, thawed_y)
    frozen_mask = rasterize(frozen_x, frozen_y)

    conflict_mask = thawed_mask & frozen_mask
    evidence_mask = thawed_mask | frozen_mask

    labels = np.full((ny, nx), -1, dtype=np.int8)
    labels[frozen_mask] = 0
    labels[thawed_mask] = 1
    labels[conflict_mask] = 2

    return {
        "thawed_mask": thawed_mask,
        "frozen_mask": frozen_mask,
        "evidence_mask": evidence_mask,
        "conflict_mask": conflict_mask,
        "labels": labels,
    }