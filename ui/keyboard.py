from __future__ import annotations

from typing import Optional, Sequence

import pygame


class OnScreenKeyboard:
    """Отрисовка виртуальной клавиатуры."""

    def __init__(
        self,
        rect: pygame.Rect,
        letters: Sequence[str],
        font: pygame.font.Font,
        background_color=(32, 32, 32),
        border_color=(70, 70, 70),
        cell_color=(70, 70, 70),
        highlight_color=(0, 150, 255),
        text_color=(230, 230, 230),
    ) -> None:
        self.rect = pygame.Rect(rect)
        self.letters = list(letters)
        self.font = font
        self.background_color = background_color
        self.border_color = border_color
        self.cell_color = cell_color
        self.highlight_color = highlight_color
        self.text_color = text_color

    def draw(self, surface: pygame.Surface, highlighted_index: Optional[int]) -> None:
        pygame.draw.rect(surface, self.background_color, self.rect, border_radius=12)
        pygame.draw.rect(surface, self.border_color, self.rect, width=2, border_radius=12)

        if not self.letters:
            return

        cell_width = self.rect.width / len(self.letters)
        for i, letter in enumerate(self.letters):
            left = self.rect.x + int(i * cell_width)
            right = self.rect.x + int((i + 1) * cell_width)
            cell_rect = pygame.Rect(left, self.rect.y, right - left, self.rect.height)
            color = self.highlight_color if highlighted_index is not None and i == highlighted_index else self.cell_color
            pygame.draw.rect(surface, color, cell_rect)
            pygame.draw.rect(surface, self.border_color, cell_rect, width=1)

            text_surface = self.font.render(letter, True, self.text_color)
            text_rect = text_surface.get_rect(center=cell_rect.center)
            surface.blit(text_surface, text_rect)
