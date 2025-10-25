from typing import Optional, Tuple

import cv2
import numpy as np


_EYE_CASCADE: Optional[cv2.CascadeClassifier] = None


def _load_eye_cascade() -> Optional[cv2.CascadeClassifier]:
    global _EYE_CASCADE
    if _EYE_CASCADE is not None:
        return _EYE_CASCADE if not _EYE_CASCADE.empty() else None

    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        _EYE_CASCADE = cv2.CascadeClassifier(cascade_path)
    except Exception:  # pragma: no cover - каскады могут отсутствовать в окружении
        _EYE_CASCADE = cv2.CascadeClassifier()

    return _EYE_CASCADE if not _EYE_CASCADE.empty() else None


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

    eye_region = _locate_eye_region(gray, prev_detection, roi_size)
    if eye_region is not None:
        ex, ey, ew, eh = eye_region
        ex1 = max(0, ex)
        ey1 = max(0, ey)
        ex2 = min(frame.shape[1], ex + ew)
        ey2 = min(frame.shape[0], ey + eh)
        if ex2 - ex1 > 10 and ey2 - ey1 > 10:
            eye_roi = enhanced[ey1:ey2, ex1:ex2]
            local_detection, confidence = _detect_from_gray(eye_roi)
            if local_detection is not None:
                lx, ly = local_detection
                best_detection = (lx + ex1, ly + ey1)
                best_confidence = confidence + 0.1

    if best_detection is None and prev_detection is not None and roi_size > 0:
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
                best_confidence = confidence + 0.05

    full_detection, full_confidence = _detect_from_gray(enhanced)
    if full_detection is not None and full_confidence >= best_confidence:
        best_detection = full_detection
        best_confidence = full_confidence

    if best_detection is None:
        fallback_detection = _fallback_threshold(gray)
        if fallback_detection is not None:
            best_detection = fallback_detection
            best_confidence = max(best_confidence, 0.15)

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

    best_detection: Optional[Tuple[int, int]] = None
    best_score = 0.0
    best_confidence = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 20:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius < 3 or radius > 90:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = float((4 * np.pi * area) / (perimeter * perimeter))
        if circularity < 0.35:
            continue

        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_intensity = cv2.mean(gray, mask=mask)[0]
        darkness = 255.0 - mean_intensity

        score = darkness * (circularity ** 2)
        if score <= best_score:
            continue

        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        best_score = score
        best_detection = (cx, cy)
        norm_darkness = max(0.0, min(1.0, darkness / 255.0))
        best_confidence = float(max(0.0, min(1.0, norm_darkness * circularity)))

    refined_detection, refined_confidence = _dark_region_centroid(gray)
    if refined_detection is not None:
        if best_detection is None:
            best_detection = refined_detection
            best_confidence = refined_confidence
        else:
            rx, ry = refined_detection
            bx, by = best_detection
            if abs(rx - bx) + abs(ry - by) <= 20 or refined_confidence > best_confidence:
                best_detection = (int((rx + bx) / 2), int((ry + by) / 2))
                best_confidence = max(best_confidence, refined_confidence)

    if best_detection is None:
        return None, 0.0

    return best_detection, best_confidence


def _fallback_threshold(gray: np.ndarray) -> Optional[Tuple[int, int]]:
    """Более мягкий бинарный поиск зрачка, близкий к первоначальной версии."""

    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(contour)
    if radius < 3 or radius > 90:
        return None
    return int(x), int(y)


def _locate_eye_region(
    gray: np.ndarray,
    prev_detection: Optional[Tuple[int, int]],
    roi_size: int,
) -> Optional[Tuple[int, int, int, int]]:
    cascade = _load_eye_cascade()
    if cascade is None:
        if prev_detection is None or roi_size <= 0:
            return None
        x, y = prev_detection
        half = roi_size // 2
        return x - half, y - half, roi_size, roi_size

    detections = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(40, 40),
    )

    if len(detections) == 0:
        if prev_detection is None or roi_size <= 0:
            return None
        x, y = prev_detection
        half = roi_size // 2
        return x - half, y - half, roi_size, roi_size

    best_region: Optional[Tuple[int, int, int, int]] = None
    best_score = float("-inf")

    for (ex, ey, ew, eh) in detections:
        region_score = _score_eye_candidate((ex, ey, ew, eh), prev_detection, gray.shape)
        if region_score > best_score:
            best_score = region_score
            best_region = (ex, ey, ew, eh)

    if best_region is None and prev_detection is not None:
        x, y = prev_detection
        half = roi_size // 2
        return x - half, y - half, roi_size, roi_size

    if best_region is None:
        return None

    ex, ey, ew, eh = best_region
    margin = int(min(ew, eh) * 0.2)
    return ex - margin, ey - margin, ew + margin * 2, eh + margin * 2


def _score_eye_candidate(
    candidate: Tuple[int, int, int, int],
    prev_detection: Optional[Tuple[int, int]],
    image_shape: Tuple[int, int],
) -> float:
    x, y, w, h = candidate
    score = float(w * h)

    height, width = image_shape
    vertical_center = height / 2.0
    horizontal_center = width / 2.0

    score -= abs((y + h / 2.0) - vertical_center) * 0.6
    score -= abs((x + w / 2.0) - horizontal_center) * 0.3

    if prev_detection is not None:
        px, py = prev_detection
        if x <= px <= x + w and y <= py <= y + h:
            score += w * h
        else:
            score -= np.hypot(px - (x + w / 2.0), py - (y + h / 2.0)) * 0.5

    return score


def _dark_region_centroid(gray: np.ndarray) -> Tuple[Optional[Tuple[int, int]], float]:
    if gray.size == 0:
        return None, 0.0

    blurred = cv2.GaussianBlur(gray, (9, 9), 1.5)
    min_val = float(np.min(blurred))
    max_val = float(np.max(blurred))
    if max_val - min_val < 15.0:
        return None, 0.0

    threshold = min_val + (max_val - min_val) * 0.4
    _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    coords = np.column_stack(np.where(mask > 0))
    if coords.size == 0:
        return None, 0.0

    weights = (255.0 - blurred[mask > 0]).astype(np.float32)
    total_weight = float(np.sum(weights))
    if total_weight <= 1e-3:
        return None, 0.0

    ys = coords[:, 0].astype(np.float32)
    xs = coords[:, 1].astype(np.float32)
    cy = float(np.sum(ys * weights) / total_weight)
    cx = float(np.sum(xs * weights) / total_weight)

    dispersion = float(np.std(xs)) + float(np.std(ys))
    normalized_dispersion = 1.0 - min(1.0, dispersion / 50.0)
    confidence = max(0.1, min(1.0, (total_weight / (mask.size * 255.0)) + normalized_dispersion * 0.5))

    return (int(round(cx)), int(round(cy))), confidence
