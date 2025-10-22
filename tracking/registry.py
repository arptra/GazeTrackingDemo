from __future__ import annotations

from typing import Dict, List, Sequence, Type

from .base import GazeTrackingAlgorithm


class TrackingAlgorithmRegistry:
    """Регистрирует и создает экземпляры алгоритмов отслеживания взгляда."""

    def __init__(self) -> None:
        self._algorithms: Dict[str, Type[GazeTrackingAlgorithm]] = {}
        self._order: List[str] = []

    def register(self, algorithm_cls: Type[GazeTrackingAlgorithm]) -> None:
        name = getattr(algorithm_cls, "name", None)
        if not name:
            raise ValueError("Алгоритм должен иметь атрибут 'name'")
        if name in self._algorithms:
            raise ValueError(f"Алгоритм '{name}' уже зарегистрирован")
        self._algorithms[name] = algorithm_cls
        self._order.append(name)

    def algorithm_names(self) -> Sequence[str]:
        return list(self._order)

    def create(self, name: str) -> GazeTrackingAlgorithm:
        if name not in self._algorithms:
            raise KeyError(f"Алгоритм '{name}' не зарегистрирован")
        return self._algorithms[name]()

    def create_by_index(self, index: int) -> GazeTrackingAlgorithm:
        if not self._order:
            raise LookupError("Нет зарегистрированных алгоритмов")
        name = self._order[index % len(self._order)]
        return self.create(name)
