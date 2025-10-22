from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from utils.pupil_detector import detect_pupil
from utils.smoothing import ExponentialSmoother

from ..base import GazeTrackingAlgorithm


class SimplePupilTracker(GazeTrackingAlgorithm):
    """Простой алгоритм: находит зрачок и сглаживает координату X."""

    name = "Простой"

    def __init__(
        self,
        smooth_alpha: float = 0.85,
        max_missed_frames: int = 10,
    ) -> None:
        self._smoother = ExponentialSmoother(smooth_alpha)
        self._last_detection: Optional[Tuple[int, int]] = None
        self._missed_frames = 0
        self._max_missed_frames = max_missed_frames

    def reset(self) -> None:
        self._smoother.reset()
        self._last_detection = None
        self._missed_frames = 0

    def process_frame(self, frame: np.ndarray) -> Optional[float]:
        detection = detect_pupil(frame)
        if detection is None:
            self._missed_frames += 1
            if self._missed_frames > self._max_missed_frames:
                self.reset()
                return None
            return self._smoother.value

        self._missed_frames = 0
        self._last_detection = detection
        norm_x = detection[0] / frame.shape[1]
        norm_x = max(0.0, min(1.0, norm_x))
        return self._smoother.update(norm_x)

    def last_detection(self) -> Optional[Tuple[int, int]]:
        return self._last_detection
