import time
from typing import Optional, Sequence, Tuple

import cv2
import pygame

from devices.cameras import discover_cameras
from tracking.algorithms.simple import SimplePupilTracker
from tracking.registry import TrackingAlgorithmRegistry
from ui.camera_interface import CameraInterface
from ui.keyboard import OnScreenKeyboard

DWELL_TIME = 2.0
LETTERS: Sequence[str] = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
WINDOW_SIZE = (1280, 720)
KEYBOARD_RECT = pygame.Rect(20, 520, WINDOW_SIZE[0] - 40, WINDOW_SIZE[1] - 540)


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Gaze Tracking Demo")

    window = pygame.display.set_mode(WINDOW_SIZE)
    keyboard_font = pygame.font.SysFont("Arial", 48)
    button_font = pygame.font.SysFont("Arial", 24)
    info_font = pygame.font.SysFont("Arial", 20)

    keyboard = OnScreenKeyboard(KEYBOARD_RECT, LETTERS, keyboard_font)

    registry = TrackingAlgorithmRegistry()
    registry.register(SimplePupilTracker)
    algorithm_names = list(registry.algorithm_names())
    if not algorithm_names:
        raise RuntimeError("Не зарегистрировано ни одного алгоритма отслеживания")
    current_algorithm_index = 0
    current_algorithm = registry.create_by_index(current_algorithm_index)

    cap: Optional[cv2.VideoCapture] = None
    selected_camera: Optional[int] = None
    tracking_enabled = False
    prev_idx: Optional[int] = None
    last_time = time.time()

    interface: Optional[CameraInterface] = None

    def select_camera(camera_index: int) -> None:
        nonlocal cap, selected_camera, tracking_enabled, prev_idx, last_time
        new_cap = cv2.VideoCapture(camera_index)
        if not new_cap.isOpened():
            new_cap.release()
            print(f"Не удалось открыть камеру {camera_index}")
            return
        new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if cap is not None:
            cap.release()
        cap = new_cap
        selected_camera = camera_index
        tracking_enabled = True
        current_algorithm.reset()
        prev_idx = None
        last_time = time.time()
        if interface is not None:
            interface.set_selected_camera(camera_index)
            interface.set_tracking_enabled(tracking_enabled)

    def toggle_tracking() -> None:
        nonlocal tracking_enabled, prev_idx, last_time
        if selected_camera is None:
            return
        tracking_enabled = not tracking_enabled
        current_algorithm.reset()
        prev_idx = None
        last_time = time.time()
        if interface is not None:
            interface.set_tracking_enabled(tracking_enabled)

    def cycle_algorithm() -> None:
        nonlocal current_algorithm_index, current_algorithm, prev_idx, last_time
        if not algorithm_names:
            return
        current_algorithm_index = (current_algorithm_index + 1) % len(algorithm_names)
        current_algorithm = registry.create_by_index(current_algorithm_index)
        current_algorithm.reset()
        prev_idx = None
        last_time = time.time()
        if interface is not None:
            interface.set_algorithm_name(current_algorithm.name, len(algorithm_names) > 1)

    interface = CameraInterface(
        WINDOW_SIZE,
        keyboard,
        on_select_camera=select_camera,
        on_toggle_tracking=toggle_tracking,
        on_cycle_algorithm=cycle_algorithm,
        button_font=button_font,
        info_font=info_font,
    )

    available_cameras = discover_cameras()
    interface.set_available_cameras(available_cameras)
    interface.set_selected_camera(selected_camera)
    interface.set_tracking_enabled(tracking_enabled)
    interface.set_algorithm_name(current_algorithm.name, len(algorithm_names) > 1)

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            else:
                interface.handle_event(event)

        frame = None
        frame_message: Optional[str] = None
        pupil_point: Optional[Tuple[int, int]] = None
        highlighted_index: Optional[int] = None

        if cap is not None and cap.isOpened():
            ret, current_frame = cap.read()
            if ret:
                frame = current_frame
                normalized_x = current_algorithm.process_frame(frame)
                pupil_point = current_algorithm.last_detection()
                if tracking_enabled:
                    if normalized_x is not None:
                        highlighted_index = int(normalized_x * len(LETTERS))
                        highlighted_index = max(0, min(len(LETTERS) - 1, highlighted_index))
                    else:
                        current_algorithm.reset()
                        prev_idx = None
                        last_time = time.time()
            else:
                frame_message = "Нет сигнала с камеры"
        else:
            frame_message = "Выберите камеру" if available_cameras else "Камеры не найдены"

        interface.draw(window, frame, frame_message, pupil_point, highlighted_index)

        if tracking_enabled and highlighted_index is not None:
            if highlighted_index == prev_idx and prev_idx is not None:
                if time.time() - last_time > DWELL_TIME:
                    print("Selected:", LETTERS[highlighted_index])
                    last_time = time.time()
            else:
                prev_idx = highlighted_index
                last_time = time.time()
        else:
            prev_idx = None

        pygame.display.flip()
        clock.tick(30)

    if cap is not None:
        cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
