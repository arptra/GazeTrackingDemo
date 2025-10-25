from __future__ import annotations

from typing import Optional, Tuple

import cv2

from ..base import GazeTrackingAlgorithm


class SimplePupilTracker(GazeTrackingAlgorithm):
    """Простой алгоритм: находит зрачок и сглаживает координату X."""

    name = "Простой"

    def __init__(
        self,
        roi: Optional[Tuple[float, float, float, float]] = None,
        smooth_alpha: float = 0.9,
        debug: bool = False,
    ) -> None:
        self._raw_roi = roi
        self._alpha = max(0.0, min(0.9999, smooth_alpha))
        self.debug = debug
        self._smooth_point: Optional[Tuple[float, float]] = None
        self._last_detection: Optional[Tuple[int, int]] = None
        self._last_norm: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        self._smooth_point = None
        self._last_detection = None
        self._last_norm = None

    # ------------------------------------------------------------------
    def process_frame(self, frame) -> Optional[float]:
        """Обрабатывает кадр и возвращает нормализованную координату X."""

        if frame is None:
            return self._last_norm[0] if self._last_norm else None

        detection = self.update(frame)
        if detection is None:
            return self._last_norm[0] if self._last_norm else None

        norm = self.get_normalized(frame.shape)
        return norm[0] if norm is not None else None

    def last_detection(self) -> Optional[Tuple[int, int]]:
        return self._last_detection

    def last_normalized(self) -> Optional[Tuple[float, float]]:
        return self._last_norm

    # ------------------------------------------------------------------
    def set_roi(self, roi: Optional[Tuple[float, float, float, float]]) -> None:
        """Позволяет вручную задать область интереса (x0, y0, x1, y1)."""

        self._raw_roi = roi

    def update(self, frame) -> Optional[Tuple[int, int]]:
        """Находит центр самого тёмного пятна и применяет сглаживание."""

        if frame is None or frame.size == 0:
            return None

        x0, y0, x1, y1 = self._resolve_roi(frame)
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        if roi.ndim == 2:
            gray = roi
        else:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        _min_val, _max_val, min_loc, _max_loc = cv2.minMaxLoc(gray)
        rx, ry = min_loc
        cx = x0 + rx
        cy = y0 + ry

        smooth = self._apply_smoothing(float(cx), float(cy))
        if smooth is None:
            return None

        sx, sy = smooth
        self._smooth_point = (sx, sy)
        self._last_detection = (int(round(sx)), int(round(sy)))

        frame_h, frame_w = frame.shape[:2]
        nx = max(0.0, min(1.0, sx / max(frame_w, 1)))
        ny = max(0.0, min(1.0, sy / max(frame_h, 1)))
        self._last_norm = (nx, ny)

        if self.debug:
            debug_frame = frame.copy()
            if debug_frame.ndim == 2:
                debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_GRAY2BGR)
            cv2.circle(debug_frame, self._last_detection, 4, (0, 0, 255), 2)
            cv2.imshow("Dark Spot Tracker", debug_frame)
            cv2.waitKey(1)

        return self._last_detection

    def get_normalized(self, frame_shape) -> Optional[Tuple[float, float]]:
        if self._last_norm is None:
            return None
        return self._last_norm

    # ------------------------------------------------------------------
    def _resolve_roi(self, frame) -> Tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        if self._raw_roi is None:
            return 0, 0, w, h

        x0, y0, x1, y1 = self._raw_roi
        if 0.0 <= x0 <= 1.0 and 0.0 <= x1 <= 1.0:
            x0 = int(round(x0 * w))
            x1 = int(round(x1 * w))
        else:
            x0 = int(round(x0))
            x1 = int(round(x1))

        if 0.0 <= y0 <= 1.0 and 0.0 <= y1 <= 1.0:
            y0 = int(round(y0 * h))
            y1 = int(round(y1 * h))
        else:
            y0 = int(round(y0))
            y1 = int(round(y1))

        x0 = max(0, min(w - 1, x0))
        y0 = max(0, min(h - 1, y0))
        x1 = max(x0 + 1, min(w, x1))
        y1 = max(y0 + 1, min(h, y1))
        return x0, y0, x1, y1

    def _apply_smoothing(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        if self._smooth_point is None:
            return x, y

        prev_x, prev_y = self._smooth_point
        smooth_x = self._alpha * prev_x + (1.0 - self._alpha) * x
        smooth_y = self._alpha * prev_y + (1.0 - self._alpha) * y
        return smooth_x, smooth_y
