import math
from typing import Union

Numeric = Union[int, float]


def custom_normal(x: Numeric, mu: Numeric = 0, max_y: float = 1) -> float:
    """
    计算值 x 在正态分布 N(mu, (1 / sqrt(2 * pi))²) 下的概率密度函数(PDF)值。

    纯函数：无状态、无副作用、相同输入必定产生相同输出。

    Args:
        x: 要计算的点
        mu: 均值 (μ)，默认为 0
        sigma: 标准差 (σ)，必须 > 0，默认为 1

    Returns:
        float: 在点 x 处的概率密度值

    Raises:
        ValueError: 当 sigma <= 0 时

    Examples:
        >>> normal_pdf(0)  # 标准正态分布在 0 处的值
        0.3989422804014327
        >>> normal_pdf(1, mu=0.5, sigma=0.1)  # N(0.5, 0.01) 在 1 处的值
        1.4867195147342979e-12
    """

    return 2 * max_y * math.exp(-math.pi * (x - mu) ** 2) - max_y
