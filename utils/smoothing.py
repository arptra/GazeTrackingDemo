from collections import deque
from statistics import median
from typing import Deque, Optional


class ExponentialSmoother:
    """Экспоненциальное сглаживание координат."""

    def __init__(self, alpha: float = 0.85):
        self.alpha = alpha
        self._value: Optional[float] = None

    def reset(self) -> None:
        """Сбрасывает накопленное значение."""

        self._value = None

    def update(self, new_value: float) -> float:
        if self._value is None:
            self._value = new_value
        else:
            self._value = self.alpha * self._value + (1 - self.alpha) * new_value
        return self._value

    def override(self, value: float) -> float:
        """Принудительно устанавливает значение сглаживателя."""

        self._value = value
        return self._value

    @property
    def value(self) -> Optional[float]:
        return self._value


class MedianSmoother:
    """Скользящая медиана для подавления выбросов."""

    def __init__(self, window_size: int = 5):
        if window_size < 1:
            raise ValueError("Размер окна медианы должен быть положительным")
        self.window_size = window_size
        self._values: Deque[float] = deque(maxlen=window_size)

    def reset(self) -> None:
        self._values.clear()

    def update(self, new_value: float) -> float:
        self._values.append(float(new_value))
        return float(median(self._values))

    @property
    def value(self) -> Optional[float]:
        if not self._values:
            return None
        return float(median(self._values))
