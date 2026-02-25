import math
from typing import Union

Numeric = Union[int, float]

def normal_pdf(x: Numeric, mu: Numeric = 0, sigma: Numeric = 1) -> float:
    """
    计算值 x 在正态分布 N(mu, sigma²) 下的概率密度函数(PDF)值。
    
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
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    
    # 避免重复计算
    two_sigma_sq = 2 * sigma * sigma
    sqrt_two_pi = math.sqrt(2 * math.pi)
    
    coefficient = 1 / (sigma * sqrt_two_pi)
    exponent = -((x - mu) ** 2) / two_sigma_sq
    
    return coefficient * math.exp(exponent)


# 使用示例
# if __name__ == "__main__":
#     # 标准正态分布 N(0, 1)
#     print(f"N(0,1) at x=0: {normal_pdf(0)}")
#     print(f"N(0,1) at x=1: {normal_pdf(1)}")
#     print(f"N(0,1) at x=-1: {normal_pdf(-1)}")
    
#     # 自定义正态分布 N(0.5, 0.1²)
#     print(f"\nN(0.5, 0.01) at x=0.5: {normal_pdf(0.5, mu=0.5, sigma=0.1)}")
#     print(f"N(0.5, 0.01) at x=0.4: {normal_pdf