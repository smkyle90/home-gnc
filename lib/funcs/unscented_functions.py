import numpy as np
import scipy
import scipy.linalg  # SciPy Linear Algebra Library


def get_sigma_points(mu, cov, default_cov=10.0):
    """Get sigma points for unscented transform. As per:

    https://en.wikipedia.org/wiki/Unscented_transform


    Args:
        mu (np.ndarray): measurement we are converting. An ns x 1 array, where ns is the number of states.
        cov (np.ndarray): covariance of measurement. An ns x ns array.
        default_cov (float): default value is Cholesky decomposition fails (matrix not positive semi-definite).

    Returns:
        sigma_pts (np.ndarray): sigma points of statistic's mean and covariances. An ns x 2 * ns array.

    """

    ns, __ = mu.shape
    nc, __ = cov.shape

    if ns != nc:
        raise ValueError("Mean value array must have dimension {} x {}".format(nc, 1))

    try:
        L = scipy.linalg.cholesky(cov, lower=True)
    except np.linalg.LinAlgError as e:
        print(e)
        L = default_cov * np.eye(nc)

    M_bar = np.sqrt(2) * L

    sigma_pts = np.block([mu - M_bar, mu + M_bar])

    return sigma_pts


def calculate_unscented_statistics(m_plus):
    """Perform the unscented transform. As per:

    https://en.wikipedia.org/wiki/Unscented_transform

    Args:
        mu_bar (np.ndarray): the measurement converted into the new coordinate frame. An ns_bar x 1 array, where ns_bar is the number of new states.
        m_plus(np.ndarray): the sigma points converted into the new coordinate frame. An ns_bar x 2 * ns array, where ns is the number of original states.

    Returns:
        mu_ut (np.ndarray): the unscented mean of the new states, an ns_bar x 1 array.
        cov_ut (np.ndarray): the unscented covariance of the new states, an ns_bar x ns_bar array.

    """
    __, nr = m_plus.shape

    mu_ut = np.mean(m_plus, axis=1).reshape(-1, 1)
    cov_ut = (m_plus - mu_ut).dot((m_plus - mu_ut).T) / nr

    return mu_ut, cov_ut
