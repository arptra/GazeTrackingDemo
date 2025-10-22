from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple


class GazeTrackingAlgorithm(ABC):
    """Абстрактный базовый класс для алгоритмов отслеживания взгляда."""

    name: str

    @abstractmethod
    def reset(self) -> None:
        """Сбрасывает внутреннее состояние алгоритма."""

    @abstractmethod
    def process_frame(self, frame) -> Optional[float]:
        """Обрабатывает кадр и возвращает нормализованную координату X (0..1)."""

    @abstractmethod
    def last_detection(self) -> Optional[Tuple[int, int]]:
        """Возвращает последнюю обнаруженную позицию зрачка в пикселях."""
