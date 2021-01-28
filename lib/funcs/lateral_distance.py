import numpy as np


def lateral_distance(x0, y0, xf, yf, thetaf, epislon=0.01):
    """Get the lateral distance of a point to a line defined by a desired
    pose in space.

    We get the distance of an object to a line defined by

        y = tan(thetaf) * (x - xf) + yf

    which is a line through the final location, at the desired orientation.

    We use this to calculate the perpendicular distance, d, from the line to
    the point (x0, y0). A "positive" distance, implies the point is "more clockwise"
    than the line, and a "negative" distance implies the point is "less clockwise".

    This is a necessary piece of information for the path tracking controller to know,
    as it relies on heading and lateral error.


    Args:
        x0 (float): x-coordinate of agent in space
        y0 (float): y-coordinate of agent in space
        xf (float): x-coordinate of target location in space
        yf (float): y-coordinate of target location in space
        thetaf (float): orientation of vehicle at target locations

    Returns:
        d (float): the lateral distance of the agent from the ideal path

    """

    # Check if the lines are vertical
    if np.abs(thetaf - np.pi / 2) < epislon:
        d = -(x0 - xf)
    elif np.abs(thetaf - 3 * np.pi / 2) < epislon:
        d = x0 - xf
    else:
        mf = np.tan(thetaf)

        # This is the solution to the dot product of intermediate vectors being zer0.
        # The intermediat vectors are r1 = (x0-xi, y0-yi) and r2 = (xf-xi, yf-yi).
        # We solve for xi and yi, which is the point on the line yf closest to (x0, y0).
        x_i = (mf ** 2 * xf + mf * y0 - mf * yf + x0) / (mf ** 2 + 1)
        y_i = (mf ** 2 * y0 + mf * x0 - mf * xf + yf) / (mf ** 2 + 1)

        r = np.array([[x0 - x_i], [y0 - y_i],])

        # Rotation matrix
        R = np.array(
            [[np.cos(thetaf), -np.sin(thetaf)], [np.sin(thetaf), np.cos(thetaf)],]
        )

        # We then rotate this by the desired orientation to get the lateral distance.
        # We mostly do this to get the sign.
        r_bar = R.T.dot(r)

        # The distance is the y-component of the rotated vector
        d = r_bar[1, 0]

    return d
