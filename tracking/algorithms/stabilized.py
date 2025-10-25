from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from utils.pupil_detector import detect_pupil_with_roi
from utils.smoothing import ExponentialSmoother, MedianSmoother

from ..base import GazeTrackingAlgorithm


class StabilizedPupilTracker(GazeTrackingAlgorithm):
    """Более устойчивый алгоритм с комбинированным сглаживанием."""

    name = "Стабилизированный"

    def __init__(
        self,
        smooth_alpha: float = 0.75,
        median_window: int = 5,
        roi_size: int = 160,
        max_step: float = 0.08,
        min_confidence: float = 0.3,
        max_missed: int = 6,
    ) -> None:
        self._roi_size = roi_size
        self._max_step = max_step
        self._min_confidence = min_confidence
        self._max_missed = max_missed
        self._exp_smoother = ExponentialSmoother(smooth_alpha)
        self._median_smoother = MedianSmoother(median_window)
        self._last_detection: Optional[Tuple[int, int]] = None
        self._stable_value: Optional[float] = None
        self._missed_frames = 0

    def reset(self) -> None:
        self._exp_smoother.reset()
        self._median_smoother.reset()
        self._last_detection = None
        self._stable_value = None
        self._missed_frames = 0

    def process_frame(self, frame: np.ndarray) -> Optional[float]:
        detection, confidence = detect_pupil_with_roi(
            frame, self._last_detection, roi_size=self._roi_size
        )

        if detection is None:
            self._missed_frames += 1
            if self._missed_frames > self._max_missed:
                self.reset()
                return None
            return self._stable_value

        self._missed_frames = 0
        self._last_detection = detection

        norm_x = detection[0] / frame.shape[1]
        norm_x = max(0.0, min(1.0, norm_x))
        median_value = self._median_smoother.update(norm_x)
        smoothed_value = self._exp_smoother.update(median_value)

        if self._stable_value is not None:
            delta = smoothed_value - self._stable_value
            if abs(delta) > self._max_step and confidence < self._min_confidence:
                direction = float(np.sign(delta))
                if direction == 0.0:
                    capped = self._stable_value
                else:
                    capped = self._stable_value + direction * self._max_step
                capped = max(0.0, min(1.0, capped))
                smoothed_value = self._exp_smoother.override(capped)

        self._stable_value = smoothed_value
        return self._stable_value

    def last_detection(self) -> Optional[Tuple[int, int]]:
        return self._last_detection
