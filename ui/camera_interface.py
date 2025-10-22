from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import pygame

from .keyboard import OnScreenKeyboard

BACKGROUND_COLOR = (18, 18, 18)
PANEL_COLOR = (32, 32, 32)
BORDER_COLOR = (70, 70, 70)
BUTTON_COLOR = (60, 60, 60)
BUTTON_HOVER_COLOR = (90, 90, 90)
BUTTON_SELECTED_COLOR = (0, 115, 210)
BUTTON_DISABLED_COLOR = (40, 40, 40)
TEXT_COLOR = (230, 230, 230)
DOT_COLOR = (255, 64, 64)
START_TRACKING_LABEL = "Запустить отслеживание"
STOP_TRACKING_LABEL = "Остановить отслеживание"
PANEL_PADDING = 16
BUTTON_HEIGHT = 48


class Button:
    """Простая кнопка для pygame."""

    def __init__(
        self,
        rect: pygame.Rect,
        text: str,
        font: pygame.font.Font,
        callback: Callable[[], None],
    ) -> None:
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.callback = callback
        self.visible = True
        self.enabled = True
        self.selected = False

    def draw(self, surface: pygame.Surface) -> None:
        if not self.visible:
            return
        mouse_pos = pygame.mouse.get_pos()
        if not self.enabled:
            color = BUTTON_DISABLED_COLOR
        elif self.selected:
            color = BUTTON_SELECTED_COLOR
        elif self.rect.collidepoint(mouse_pos):
            color = BUTTON_HOVER_COLOR
        else:
            color = BUTTON_COLOR

        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, self.rect, width=2, border_radius=8)

        text_surface = self.font.render(self.text, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def handle_event(self, event: pygame.event.Event) -> None:
        if (
            self.visible
            and self.enabled
            and event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.rect.collidepoint(event.pos)
        ):
            self.callback()


class CameraInterface:
    """Управляет отрисовкой интерфейса камеры и панелей управления."""

    def __init__(
        self,
        window_size: Tuple[int, int],
        keyboard: OnScreenKeyboard,
        on_select_camera: Callable[[int], None],
        on_toggle_tracking: Callable[[], None],
        on_cycle_algorithm: Callable[[], None],
        button_font: pygame.font.Font,
        info_font: pygame.font.Font,
    ) -> None:
        self.window_width, self.window_height = window_size
        self.keyboard = keyboard
        self.on_select_camera = on_select_camera
        self.on_toggle_tracking = on_toggle_tracking
        self.on_cycle_algorithm = on_cycle_algorithm
        self.button_font = button_font
        self.info_font = info_font

        self.camera_rect = pygame.Rect(20, 20, 640, 480)
        self.button_panel_rect = pygame.Rect(
            self.camera_rect.right + 20,
            self.camera_rect.top,
            260,
            self.camera_rect.height,
        )

        self.camera_buttons: List[Button] = []
        self.static_buttons: List[Button] = []
        self.start_button = Button(
            pygame.Rect(0, 0, 0, BUTTON_HEIGHT),
            START_TRACKING_LABEL,
            self.button_font,
            self.on_toggle_tracking,
        )
        self.start_button.visible = False
        self.start_button.enabled = False

        self.algorithm_button = Button(
            pygame.Rect(0, 0, 0, BUTTON_HEIGHT),
            "Сменить алгоритм",
            self.button_font,
            self.on_cycle_algorithm,
        )
        self.algorithm_button.enabled = False

        self.static_buttons.extend([self.algorithm_button, self.start_button])

        self.available_cameras: Sequence[int] = []
        self.selected_camera: Optional[int] = None
        self.tracking_enabled = False
        self.algorithm_name = "—"

    def set_available_cameras(self, cameras: Sequence[int]) -> None:
        self.available_cameras = list(cameras)
        self._rebuild_camera_buttons()

    def set_selected_camera(self, camera_index: Optional[int]) -> None:
        self.selected_camera = camera_index
        for button in self.camera_buttons:
            button.selected = getattr(button, "camera_index", None) == camera_index
        self.start_button.visible = camera_index is not None
        self.start_button.enabled = camera_index is not None

    def set_tracking_enabled(self, enabled: bool) -> None:
        self.tracking_enabled = enabled
        self.start_button.text = STOP_TRACKING_LABEL if enabled else START_TRACKING_LABEL

    def set_algorithm_name(self, name: str, has_multiple: bool) -> None:
        self.algorithm_name = name
        self.algorithm_button.text = (
            "Сменить алгоритм" if has_multiple else f"Алгоритм: {name}"
        )
        self.algorithm_button.enabled = has_multiple

    def handle_event(self, event: pygame.event.Event) -> None:
        for button in self.static_buttons:
            button.handle_event(event)
        for button in self.camera_buttons:
            button.handle_event(event)

    def draw(
        self,
        surface: pygame.Surface,
        frame: Optional["cv2.Mat"],
        frame_message: Optional[str],
        pupil_point: Optional[Tuple[int, int]],
        highlighted_index: Optional[int],
    ) -> None:
        surface.fill(BACKGROUND_COLOR)

        pygame.draw.rect(surface, PANEL_COLOR, self.camera_rect, border_radius=12)
        pygame.draw.rect(surface, BORDER_COLOR, self.camera_rect, width=2, border_radius=12)

        if frame is not None:
            frame_surface = self._create_frame_surface(frame)
            surface.blit(frame_surface, self.camera_rect.topleft)
        if frame_message:
            message_surface = self.info_font.render(frame_message, True, TEXT_COLOR)
            message_rect = message_surface.get_rect(center=self.camera_rect.center)
            surface.blit(message_surface, message_rect)

        if frame is not None and pupil_point is not None:
            self._draw_pupil(surface, frame, pupil_point)

        self._draw_button_panel(surface)
        self.keyboard.draw(surface, highlighted_index if self.tracking_enabled else None)

    def _rebuild_camera_buttons(self) -> None:
        self.camera_buttons.clear()
        for camera_index in self.available_cameras:
            rect = pygame.Rect(0, 0, 0, BUTTON_HEIGHT)
            button = Button(
                rect,
                f"Камера {camera_index}",
                self.button_font,
                lambda i=camera_index: self.on_select_camera(i),
            )
            setattr(button, "camera_index", camera_index)
            self.camera_buttons.append(button)
        self.set_selected_camera(self.selected_camera)

    def _draw_button_panel(self, surface: pygame.Surface) -> None:
        self._layout_controls()
        pygame.draw.rect(surface, PANEL_COLOR, self.button_panel_rect, border_radius=12)
        pygame.draw.rect(surface, BORDER_COLOR, self.button_panel_rect, width=2, border_radius=12)

        title = self.info_font.render("Доступные камеры", True, TEXT_COLOR)
        surface.blit(title, self._title_position)

        if self._status_position is not None and self.selected_camera is not None:
            status_text = self.info_font.render(
                f"Активная: {self.selected_camera}", True, TEXT_COLOR
            )
            surface.blit(status_text, self._status_position)

        algo_label = self.info_font.render(
            f"Текущий алгоритм: {self.algorithm_name}", True, TEXT_COLOR
        )
        surface.blit(algo_label, self._algorithm_label_position)

        for button in self.static_buttons:
            button.draw(surface)

        if not self.available_cameras:
            empty_text = self.info_font.render("Камеры не найдены", True, TEXT_COLOR)
            surface.blit(empty_text, (self._inner_left, self._camera_list_top))
        else:
            y = self._camera_list_top
            for button in self.camera_buttons:
                button.rect.update(self._inner_left, y, self._inner_width, BUTTON_HEIGHT)
                button.draw(surface)
                y += BUTTON_HEIGHT + 12

    def _create_frame_surface(self, frame) -> pygame.Surface:
        resized = cv2.resize(frame, (self.camera_rect.width, self.camera_rect.height))
        frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        surface = pygame.image.frombuffer(
            frame_rgb.tobytes(),
            (self.camera_rect.width, self.camera_rect.height),
            "RGB",
        )
        return surface.convert()

    def _draw_pupil(
        self,
        surface: pygame.Surface,
        frame,
        pupil_point: Tuple[int, int],
    ) -> None:
        scale_x = self.camera_rect.width / frame.shape[1]
        scale_y = self.camera_rect.height / frame.shape[0]
        dot_position = (
            self.camera_rect.x + int(pupil_point[0] * scale_x),
            self.camera_rect.y + int(pupil_point[1] * scale_y),
        )
        pygame.draw.circle(surface, DOT_COLOR, dot_position, 8)

    def _layout_controls(self) -> None:
        inner_left = self.button_panel_rect.x + PANEL_PADDING
        inner_width = self.button_panel_rect.width - 2 * PANEL_PADDING
        line_height = self.info_font.get_linesize()

        y = self.button_panel_rect.y + PANEL_PADDING
        self._title_position = (inner_left, y)
        y += line_height + 6

        if self.selected_camera is not None:
            self._status_position = (inner_left, y)
            y += line_height + 12
        else:
            self._status_position = None

        self._algorithm_label_position = (inner_left, y)
        y += line_height + 8

        self.algorithm_button.rect.update(inner_left, y, inner_width, BUTTON_HEIGHT)
        y = self.algorithm_button.rect.bottom + 24

        self._camera_list_top = y
        self._inner_left = inner_left
        self._inner_width = inner_width

        start_y = self.button_panel_rect.bottom - PANEL_PADDING - BUTTON_HEIGHT
        self.start_button.rect.update(inner_left, start_y, inner_width, BUTTON_HEIGHT)
