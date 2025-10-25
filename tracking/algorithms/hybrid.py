from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from ..base import GazeTrackingAlgorithm


@dataclass
class _PupilDetection:
    center: Tuple[int, int]
    radius: float
    mask: np.ndarray
    roi: np.ndarray


class HybridPupilTracker(GazeTrackingAlgorithm):
    """Совмещает детекцию и трекинг зрачка для устойчивых координат."""

    name = "Hybrid IR"

    def __init__(
        self,
        roi_size: int = 100,
        smooth_alpha: float = 0.9,
        dead_zone_ratio: float = 0.02,
        min_radius: int = 5,
        max_radius: int = 60,
        debug: bool = False,
    ) -> None:
        self._roi_size = int(max(20, roi_size))
        self._alpha = float(np.clip(smooth_alpha, 0.0, 0.9999))
        self._dead_zone_ratio = float(max(0.0, dead_zone_ratio))
        self._min_radius = float(min_radius)
        self._max_radius = float(max_radius)
        self.debug = debug

        self._tracker = None
        self._last_detection: Optional[Tuple[int, int]] = None
        self._last_norm: Optional[Tuple[float, float]] = None
        self._smooth_point: Optional[np.ndarray] = None
        self._active_roi: Optional[Tuple[int, int, int, int]] = None
        self._last_mask: Optional[np.ndarray] = None
        self._last_roi_frame: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._tracker = None
        self._last_detection = None
        self._last_norm = None
        self._smooth_point = None
        self._active_roi = None
        self._last_mask = None
        self._last_roi_frame = None

    def process_frame(self, frame: np.ndarray) -> Optional[float]:
        if frame is None or frame.size == 0:
            return self._last_norm[0] if self._last_norm else None

        gray = self._ensure_gray(frame)
        detection = None

        if self._tracker is not None and self._active_roi is not None:
            success, box = self._tracker.update(frame)
            if success:
                x, y, w, h = box
                cx = int(x + w / 2)
                cy = int(y + h / 2)
                roi_bbox = (int(x), int(y), int(w), int(h))
                detection = _PupilDetection(
                    (cx, cy),
                    max(w, h) / 2,
                    self._last_mask if self._last_mask is not None else np.zeros_like(gray),
                    self._extract_roi_frame(frame, roi_bbox),
                )
                self._active_roi = (int(x), int(y), int(w), int(h))
            else:
                self._tracker = None
                self._active_roi = None

        if detection is None:
            detection = self._detect_pupil(frame, gray)
            if detection is not None:
                self._initialize_tracker(frame, detection)

        if detection is None:
            return self._last_norm[0] if self._last_norm else None

        self._last_mask = detection.mask
        self._last_roi_frame = detection.roi
        smooth_point = self._apply_smoothing(np.array(detection.center, dtype=np.float32), frame.shape)
        self._smooth_point = smooth_point.copy()
        cx, cy = smooth_point
        self._last_detection = (int(round(cx)), int(round(cy)))

        frame_h, frame_w = gray.shape[:2]
        norm_x = float(np.clip(cx / max(frame_w, 1), 0.0, 1.0))
        norm_y = float(np.clip(cy / max(frame_h, 1), 0.0, 1.0))
        self._last_norm = (norm_x, norm_y)

        if self.debug:
            self._show_debug(frame, detection, smooth_point)

        return norm_x

    def last_detection(self) -> Optional[Tuple[int, int]]:
        return self._last_detection

    def last_normalized(self) -> Optional[Tuple[float, float]]:
        return self._last_norm

    # ------------------------------------------------------------------
    def _initialize_tracker(self, frame: np.ndarray, detection: _PupilDetection) -> None:
        bbox = self._build_roi_bbox(frame.shape, detection.center)
        tracker = self._create_tracker()
        if tracker is None:
            self._tracker = None
            self._active_roi = None
            return
        ok = tracker.init(frame, bbox)
        if not ok:
            self._tracker = None
            self._active_roi = None
            return
        self._tracker = tracker
        self._active_roi = bbox

    def _create_tracker(self):  # type: ignore[override]
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
        if hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
            return cv2.legacy.TrackerKCF_create()
        return None

    def _detect_pupil(self, frame: np.ndarray, gray: np.ndarray) -> Optional[_PupilDetection]:
        inpainted = self._remove_glint(gray)
        clahe = self._apply_clahe(inpainted)
        mask = self._build_mask(clahe)
        mask = self._clean_mask(mask)
        contour = self._select_contour(mask)
        if contour is None:
            return None

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius < self._min_radius or radius > self._max_radius:
            return None
        cx_i, cy_i = int(cx), int(cy)

        roi_bbox = self._build_roi_bbox(frame.shape, (cx_i, cy_i))
        roi_frame = self._extract_roi_frame(frame, roi_bbox)

        self._last_mask = mask
        self._last_roi_frame = roi_frame

        return _PupilDetection(center=(cx_i, cy_i), radius=float(radius), mask=mask, roi=roi_frame)

    def _build_roi_bbox(self, frame_shape, center: Tuple[int, int]) -> Tuple[int, int, int, int]:
        h, w = frame_shape[:2]
        size = self._roi_size
        half = size // 2
        cx, cy = center
        x0 = np.clip(cx - half, 0, max(w - size, 0))
        y0 = np.clip(cy - half, 0, max(h - size, 0))
        if x0 + size > w:
            size = w - x0
        if y0 + size > h:
            size = h - y0
        return int(x0), int(y0), int(max(size, 1)), int(max(size, 1))

    def _extract_roi_frame(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = bbox
        x1 = max(0, min(frame.shape[1], x + w))
        y1 = max(0, min(frame.shape[0], y + h))
        roi = frame[y:y1, x:x1]
        if roi.size == 0:
            return roi
        if roi.ndim == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.resize(roi, (self._roi_size, self._roi_size), interpolation=cv2.INTER_LINEAR)

    def _apply_smoothing(self, point: np.ndarray, frame_shape) -> np.ndarray:
        if self._smooth_point is None:
            return point
        frame_h, frame_w = frame_shape[:2]
        dead_x = self._dead_zone_ratio * frame_w
        dead_y = self._dead_zone_ratio * frame_h
        prev = self._smooth_point
        dx = point[0] - prev[0]
        dy = point[1] - prev[1]
        new_point = prev.copy()
        if abs(dx) >= dead_x:
            new_point[0] = self._alpha * prev[0] + (1.0 - self._alpha) * point[0]
        if abs(dy) >= dead_y:
            new_point[1] = self._alpha * prev[1] + (1.0 - self._alpha) * point[1]
        return new_point

    @staticmethod
    def _ensure_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _remove_glint(gray: np.ndarray) -> np.ndarray:
        _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        return cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)

    @staticmethod
    def _apply_clahe(gray: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    @staticmethod
    def _build_mask(gray: np.ndarray) -> np.ndarray:
        thr = np.percentile(gray, 10) + 5
        _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )
        return cv2.bitwise_or(mask, adaptive)

    @staticmethod
    def _clean_mask(mask: np.ndarray) -> np.ndarray:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return mask

    def _select_contour(self, mask: np.ndarray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = -1.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1.0:
                continue
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-5)
            if circularity < 0.6 or circularity > 1.2:
                continue
            _, radius = cv2.minEnclosingCircle(contour)
            if radius < self._min_radius or radius > self._max_radius:
                continue
            score = circularity * area
            if score > best_score:
                best_score = score
                best = contour
        return best

    def _show_debug(self, frame: np.ndarray, detection: _PupilDetection, smooth_point: np.ndarray) -> None:
        vis = frame.copy()
        cv2.circle(vis, detection.center, int(max(detection.radius, 1)), (0, 255, 0), 2)
        cv2.circle(vis, (int(smooth_point[0]), int(smooth_point[1])), 4, (0, 0, 255), -1)
        cv2.imshow("Original", vis)
        if detection.mask is not None:
            cv2.imshow("Mask", detection.mask)
        if self._last_roi_frame is not None and self._last_roi_frame.size:
            cv2.imshow("ROI", self._last_roi_frame)
        cv2.waitKey(1)
