from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from utils.pupil_detector import detect_pupil
from utils.smoothing import ExponentialSmoother

from ..base import GazeTrackingAlgorithm


class SimplePupilTracker(GazeTrackingAlgorithm):
    """Простой алгоритм: находит зрачок и сглаживает координату X."""

    name = "Простой"

    def __init__(self, smooth_alpha: float = 0.85) -> None:
        self._smooth_alpha = smooth_alpha
        self._smoother = ExponentialSmoother(smooth_alpha)
        self._last_detection: Optional[Tuple[int, int]] = None

    def reset(self) -> None:
        self._smoother = ExponentialSmoother(self._smooth_alpha)
        self._last_detection = None

    def process_frame(self, frame: np.ndarray) -> Optional[float]:
        detection = detect_pupil(frame)
        self._last_detection = detection
        if detection is None:
            self._smoother = ExponentialSmoother(self._smooth_alpha)
            return None
        norm_x = detection[0] / frame.shape[1]
        return self._smoother.update(norm_x)

    def last_detection(self) -> Optional[Tuple[int, int]]:
        return self._last_detection
