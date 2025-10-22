from typing import Optional, Tuple

import cv2
import numpy as np


def detect_pupil(frame: np.ndarray) -> Optional[Tuple[int, int]]:
    """Возвращает (x, y) центра зрачка или None."""

    detection, _ = detect_pupil_with_roi(frame, None)
    return detection


def detect_pupil_with_roi(
    frame: np.ndarray,
    prev_detection: Optional[Tuple[int, int]],
    roi_size: int = 160,
) -> Tuple[Optional[Tuple[int, int]], float]:
    """Более устойчивое обнаружение зрачка.

    Возвращает кортеж (координаты или None, confidence).
    Если передана предыдущая позиция, сначала выполняется поиск в ограниченной области.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    best_detection: Optional[Tuple[int, int]] = None
    best_confidence = 0.0

    if prev_detection is not None and roi_size > 0:
        x, y = prev_detection
        half = roi_size // 2
        x0 = max(0, x - half)
        y0 = max(0, y - half)
        x1 = min(frame.shape[1], x + half)
        y1 = min(frame.shape[0], y + half)
        if x1 - x0 > 10 and y1 - y0 > 10:
            roi = enhanced[y0:y1, x0:x1]
            local_detection, confidence = _detect_from_gray(roi)
            if local_detection is not None:
                lx, ly = local_detection
                best_detection = (lx + x0, ly + y0)
                best_confidence = confidence + 0.05  # небольшой бонус за согласованность

    full_detection, full_confidence = _detect_from_gray(enhanced)
    if full_detection is not None and full_confidence >= best_confidence:
        best_detection = full_detection
        best_confidence = full_confidence

    return best_detection, float(best_confidence)


def _detect_from_gray(gray: np.ndarray) -> Tuple[Optional[Tuple[int, int]], float]:
    """Обнаружение зрачка на одном канале изображения."""

    if gray.size == 0:
        return None, 0.0

    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, thresh = cv2.threshold(
        normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < 30:
        return None, 0.0

    (x, y), radius = cv2.minEnclosingCircle(contour)
    if radius < 4 or radius > 80:
        return None, 0.0

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return None, 0.0
    circularity = float((4 * np.pi * area) / (perimeter * perimeter))
    confidence = max(0.0, min(1.0, circularity))

    return (int(x), int(y)), confidence
