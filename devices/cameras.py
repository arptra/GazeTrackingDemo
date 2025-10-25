from typing import List

import cv2


def discover_cameras(max_devices: int = 6, max_consecutive_failures: int = 3) -> List[int]:
    """Определяет доступные индексы подключенных камер."""

    indices: List[int] = []
    failures = 0
    for index in range(max_devices):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            indices.append(index)
            failures = 0
        else:
            failures += 1
            if failures >= max_consecutive_failures:
                cap.release()
                break
        cap.release()
    return indices
