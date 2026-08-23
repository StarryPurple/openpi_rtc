import numpy as np


def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float32):
    """Pulls out x, y, and z columns from the cloud recordarray, and returns
    a 3xN matrix.
    """
    mask = None
    # remove crap points
    if remove_nans:
        mask = (
            np.isfinite(cloud_array["x"])
            & np.isfinite(cloud_array["y"])
            & np.isfinite(cloud_array["z"])
        )
        cloud_array = cloud_array[mask]

    # pull out x, y, and z values
    points = np.zeros(cloud_array.shape + (3,), dtype=dtype)
    points[..., 0] = cloud_array["x"]
    points[..., 1] = cloud_array["y"]
    points[..., 2] = cloud_array["z"]

    return points, mask
