from pandas.tseries.frequencies import to_offset
from gluonts.time_feature import TimeFeature, norm_freq_str
from typing import List, Optional
import numpy as np
from gluonts.core.component import validated
import pandas as pd


class FourierDateFeatures(TimeFeature):
    """Fourier date features."""
    @validated()
    def __init__(self, freq: str) -> None:
        super().__init__()
        # reocurring freq
        freqs = [
            "month",
            "day",
            "hour",
            "minute",
            "weekofyear",
            "weekday",
            "dayofweek",
            "dayofyear",
            "daysinmonth",
        ]

        assert freq in freqs
        self.freq = freq

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """获取日期特征

        Args:
            index (pd.DatetimeIndex): 日期索引,shape (len(index),)

        Returns:
            np.ndarray: 日期的正弦余弦特征 shape (2, len(index))
        """

        values = getattr(index, self.freq)
        num_values = max(values) + 1
        steps = [x * 2.0 * np.pi / num_values for x in values]
        # shape (2, len(index))
        return np.vstack([np.cos(steps), np.sin(steps)])


def fourier_time_features_from_frequency(freq_str: str) -> List[TimeFeature]:
    """生成序列时间特征

    Args:
        freq_str (str): 时间序列频率 1H or 15min

    Returns:
        List[TimeFeature]: 时间特征列表 [FourierDateFeatures(freq='weekofyear'), FourierDateFeatures(freq='dayofweek'),...]
    """
    offset = to_offset(freq_str)
    granularity = norm_freq_str(offset.name)  # H or min

    features = {
        "M": ["weekofyear"],
        "W": ["daysinmonth", "weekofyear"],
        "D": ["dayofweek"],
        "B": ["dayofweek", "dayofyear"],
        "H": ["hour", "dayofweek"],
        "min": ["minute", "hour", "dayofweek"],
        "T": ["minute", "hour", "dayofweek"],
    }

    assert granularity in features, f"freq {granularity} not supported"

    feature_classes: List[TimeFeature] = [
        FourierDateFeatures(freq=freq) for freq in features[granularity]
    ]

    return feature_classes


""" lag for fourier time features"""


def lags_for_fourier_time_features_from_frequency(
    freq_str: str, num_lags: Optional[int] = None
) -> List[int]:
    """生成时间特征的lags

    Args:
        freq_str (str): 时间序列频率 1H or 15min
        num_lags (Optional[int], optional): lags 个数. Defaults to None.

    Returns:
        List[int]: 时序特征的滞后序列 [1, 24, 168] or [1, 4, 12, 24, 48]
    """
    offset = to_offset(freq_str)
    _, granularity = offset.n, offset.name

    if granularity == "M":
        lags = [[1, 12]]
    elif granularity == "D":
        lags = [[1, 7, 14]]
    elif granularity == "B":
        lags = [[1, 2]]
    elif granularity == "H":  # 1H
        lags = [[1, 24, 168]]
    elif granularity in ("T", "min"):
        lags = [[1, 4, 12, 24, 48]]
    else:
        lags = [[1]]

    # use less lags
    output_lags = list([int(lag)
                       for sub_list in lags for lag in sub_list])  # [1, 24, 168]
    output_lags = sorted(list(set(output_lags)))
    return output_lags[:num_lags]
