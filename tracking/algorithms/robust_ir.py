from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from ..base import GazeTrackingAlgorithm


@dataclass
class _DetectionResult:
    point: Tuple[int, int]
    mask: np.ndarray
    roi_origin: Tuple[int, int]
    clahe: np.ndarray


class RobustIRPupilTracker(GazeTrackingAlgorithm):
    """Устойчивый алгоритм детекции зрачка для ИК-видео."""

    name = "IR устойчивый"

    def __init__(
        self,
        smooth_alpha: float = 0.9,
        dead_zone_ratio: float = 0.02,
        max_missed_frames: int = 12,
        debug: bool = False,
    ) -> None:
        self._alpha = float(smooth_alpha)
        self._dead_zone_ratio = float(dead_zone_ratio)
        self._max_missed_frames = int(max_missed_frames)
        self.debug = debug

        self._smooth_point: Optional[np.ndarray] = None
        self._last_detection: Optional[Tuple[int, int]] = None
        self._stable_norm: Optional[float] = None
        self._missed_frames = 0

        self._roi_rect: Optional[Tuple[int, int, int, int]] = None
        self._roi_mask: Optional[np.ndarray] = None

    # --- Публичные настройки -------------------------------------------------
    def set_roi_rect(self, rect: Optional[Tuple[int, int, int, int]]) -> None:
        """Задает прямоугольную область глаза (x, y, w, h)."""

        self._roi_rect = rect

    def set_roi_mask(self, mask: Optional[np.ndarray]) -> None:
        """Позволяет передать бинарную маску области глаза."""

        if mask is None:
            self._roi_mask = None
            return
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        self._roi_mask = mask

    # --- Реализация базового интерфейса --------------------------------------
    def reset(self) -> None:
        self._smooth_point = None
        self._last_detection = None
        self._stable_norm = None
        self._missed_frames = 0

    def process_frame(self, frame: np.ndarray) -> Optional[float]:
        if frame is None or frame.size == 0:
            return self._stable_norm

        gray = self._to_gray(frame)
        detection = self._detect_pupil(gray)

        if detection is None:
            self._missed_frames += 1
            if self._missed_frames > self._max_missed_frames:
                self._smooth_point = None
                self._last_detection = None
            return self._stable_norm

        self._missed_frames = 0
        cx, cy = detection.point

        smooth_point = self._apply_smoothing(np.array([float(cx), float(cy)]), frame.shape)
        smoothed_x = float(smooth_point[0])
        smoothed_y = float(smooth_point[1])
        self._last_detection = (int(round(smoothed_x)), int(round(smoothed_y)))

        frame_width = frame.shape[1]
        norm_x = max(0.0, min(1.0, smoothed_x / max(frame_width, 1)))
        self._stable_norm = norm_x

        if self.debug:
            self._show_debug_windows(frame, detection, smooth_point)

        return norm_x

    def last_detection(self) -> Optional[Tuple[int, int]]:
        return self._last_detection

    # --- Частная логика ------------------------------------------------------
    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _detect_pupil(self, gray: np.ndarray) -> Optional[_DetectionResult]:
        roi_gray, origin = self._extract_roi(gray)
        if roi_gray.size == 0:
            return None

        inpainted = self._remove_glint(roi_gray)
        clahe = self._apply_clahe(inpainted)
        mask = self._build_binary_mask(clahe)
        mask = self._apply_morphology(mask)

        contour = self._select_contour(mask)
        if contour is None:
            return None

        cx, cy = self._contour_center(contour)
        cx += origin[0]
        cy += origin[1]

        full_mask = np.zeros_like(gray, dtype=np.uint8)
        y0, x0 = origin[1], origin[0]
        full_mask[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]] = mask

        return _DetectionResult(point=(int(round(cx)), int(round(cy))), mask=full_mask, roi_origin=origin, clahe=clahe)

    def _extract_roi(self, gray: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        h, w = gray.shape[:2]
        x0, y0, x1, y1 = 0, 0, w, h

        if self._roi_rect is not None:
            rx, ry, rw, rh = self._roi_rect
            x0 = max(0, rx)
            y0 = max(0, ry)
            x1 = min(w, rx + max(0, rw))
            y1 = min(h, ry + max(0, rh))

        if self._roi_mask is not None and self._roi_mask.shape[:2] == gray.shape[:2]:
            coords = np.column_stack(np.where(self._roi_mask > 0))
            if coords.size:
                mask_y0, mask_x0 = coords.min(axis=0)
                mask_y1, mask_x1 = coords.max(axis=0) + 1
                x0 = max(x0, int(mask_x0))
                y0 = max(y0, int(mask_y0))
                x1 = min(x1, int(mask_x1))
                y1 = min(y1, int(mask_y1))

        x0 = max(0, min(x0, w - 1))
        y0 = max(0, min(y0, h - 1))
        x1 = max(x0 + 1, min(x1, w))
        y1 = max(y0 + 1, min(y1, h))

        roi = gray[y0:y1, x0:x1]
        return roi, (x0, y0)

    @staticmethod
    def _remove_glint(gray: np.ndarray) -> np.ndarray:
        mask_glint = cv2.inRange(gray, 220, 255)
        if np.count_nonzero(mask_glint) == 0:
            return gray
        return cv2.inpaint(gray, mask_glint, 3, cv2.INPAINT_TELEA)

    @staticmethod
    def _apply_clahe(gray: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    @staticmethod
    def _build_binary_mask(gray: np.ndarray) -> np.ndarray:
        percentile = float(np.percentile(gray, 10)) if gray.size else 0.0
        thr = max(0.0, min(255.0, percentile + 5.0))
        _, thresh = cv2.threshold(gray, int(round(thr)), 255, cv2.THRESH_BINARY_INV)
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            5,
        )
        return cv2.bitwise_and(thresh, adaptive)

    @staticmethod
    def _apply_morphology(mask: np.ndarray) -> np.ndarray:
        if mask.size == 0:
            return mask
        open_kernel = np.ones((3, 3), np.uint8)
        close_kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        return cleaned

    @staticmethod
    def _select_contour(mask: np.ndarray) -> Optional[np.ndarray]:
        if mask.size == 0:
            return None
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        hierarchy = hierarchy[0] if hierarchy is not None else None

        best_score = float("-inf")
        best_contour: Optional[np.ndarray] = None

        for idx, contour in enumerate(contours):
            if hierarchy is not None and hierarchy[idx][3] != -1:
                # Пропускаем внутренние контуры (дыры)
                continue
            area = float(cv2.contourArea(contour))
            if area <= 1.0:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0.0:
                continue
            circularity = (4.0 * np.pi * area) / (perimeter * perimeter + 1e-5)
            if not 0.6 <= circularity <= 1.2:
                continue
            radius = np.sqrt(area / np.pi)
            if not 5.0 < radius < 60.0:
                continue

            score = circularity * area
            if score > best_score:
                best_score = score
                best_contour = contour

        return best_contour

    @staticmethod
    def _contour_center(contour: np.ndarray) -> Tuple[int, int]:
        moments = cv2.moments(contour)
        if moments["m00"] > 1e-6:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            return cx, cy
        (x, y), _ = cv2.minEnclosingCircle(contour)
        return int(round(x)), int(round(y))

    def _apply_smoothing(self, point: np.ndarray, frame_shape) -> np.ndarray:
        frame_height, frame_width = frame_shape[0], frame_shape[1]
        if self._smooth_point is None:
            self._smooth_point = point
            return point

        smooth = self._smooth_point
        dead_zone_x = self._dead_zone_ratio * frame_width
        dead_zone_y = self._dead_zone_ratio * frame_height

        dx = point[0] - smooth[0]
        dy = point[1] - smooth[1]

        if abs(dx) > dead_zone_x:
            smooth[0] = self._alpha * smooth[0] + (1.0 - self._alpha) * point[0]
        if abs(dy) > dead_zone_y:
            smooth[1] = self._alpha * smooth[1] + (1.0 - self._alpha) * point[1]

        self._smooth_point = smooth
        return smooth

    def _show_debug_windows(
        self,
        frame: np.ndarray,
        detection: _DetectionResult,
        smooth_point: np.ndarray,
    ) -> None:
        debug_frame = frame.copy()
        cv2.circle(debug_frame, self._last_detection, 6, (0, 255, 0), 2)
        cv2.putText(
            debug_frame,
            f"({self._last_detection[0]}, {self._last_detection[1]})",
            (self._last_detection[0] + 10, self._last_detection[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        mask_vis = detection.mask
        clahe_vis = cv2.normalize(detection.clahe, None, 0, 255, cv2.NORM_MINMAX)
        clahe_vis = cv2.cvtColor(clahe_vis, cv2.COLOR_GRAY2BGR)

        hist = cv2.calcHist([detection.clahe], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, None, alpha=0, beta=200, norm_type=cv2.NORM_MINMAX)
        hist_img = np.zeros((200, 256, 3), dtype=np.uint8)
        for x in range(256):
            value = int(round(hist[x][0]))
            cv2.line(hist_img, (x, 199), (x, 199 - value), (255, 255, 255), 1)

        cv2.imshow("Robust IR - Original", debug_frame)
        cv2.imshow("Robust IR - Mask", mask_vis)
        cv2.imshow("Robust IR - CLAHE", clahe_vis)
        cv2.imshow("Robust IR - Histogram", hist_img)
        cv2.waitKey(1)

