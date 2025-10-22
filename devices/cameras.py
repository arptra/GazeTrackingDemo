from typing import List

import cv2


def discover_cameras(max_devices: int = 6) -> List[int]:
    """Определяет доступные индексы подключенных камер."""
    indices: List[int] = []
    for index in range(max_devices):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            indices.append(index)
        cap.release()
    return indices
