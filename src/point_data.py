import math

import torch



def generate_center_line_points(
    m_mu,
    m_std,
    num_pts,
    max_x=1.0
):
    """
    Generates a 2D point set around a straing line that passes through the origin.
    Point XY-coordinates are drawn from a uniform distribution.
    Point slopes are drawn from a normal distribution.

    :param m_mu: float
        Mean of the line slope.
    :param m_std: float
        Standard deviation of the line slope.
    :param num_pts: int
        Number of points to generate.
    :param max_x: float, default=1.0
        Maximum X-coordinate value.
    
    :return: torch.Tensor
        A (N x 2) float tensor with the point coordinates.
    """

    m = torch.normal(m_mu, m_std, (num_pts,))
    d = (torch.rand(num_pts) - 0.5) * 2 * max_x

    xy = torch.stack((d, d * m)).T

    return xy



def generate_circle_points(
    x_mu,
    x_std,
    y_mu,
    y_std,
    r_mu,
    r_std,
    num_pts
):
    """
    Generates a 2D point set around a circle.
    Circle center XY-coodinates and radius are drawn from normal distributions.

    :param x_mu: float
        Mean of the circle center X-coordinate.
    :param x_std: float
        Standard deviation of the circle center X-coordinate.
    :param y_mu: float
        Mean of the circle center Y-coordinate.
    :param y_std: float
        Standard deviation of the circle center Y-coordinate.
    :param r_mu: float
        Mean of the circle radius.
    :param r_std: float
        Standard deviation of the circle radius.
    :param num_pts: int
        Number of points to generate.
    
    :return: torch.Tensor
        A (N x 2) float tensor with the point coordinates.
    """

    x = torch.normal(x_mu, x_std, (num_pts,))
    y = torch.normal(y_mu, y_std, (num_pts,))
    r = torch.normal(r_mu, r_std, (num_pts,))
    t = torch.rand((num_pts,)) * 2 * math.pi

    x += r * torch.cos(t)
    y += r * torch.sin(t)

    xy = torch.stack((x, y)).T

    return xy
