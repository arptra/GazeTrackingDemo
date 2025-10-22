import cv2
import pygame
import time
from utils.pupil_detector import detect_pupil
from utils.smoothing import ExponentialSmoother

SMOOTH_ALPHA = 0.85
DWELL_TIME = 2.0
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ ")

pygame.init()
WIDTH, HEIGHT = 1200, 200
win = pygame.display.set_mode((WIDTH, HEIGHT))
font = pygame.font.SysFont("Arial", 64)
clock = pygame.time.Clock()

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

smoother = ExponentialSmoother(SMOOTH_ALPHA)
prev_idx = None
last_time = 0

def draw_keyboard(idx):
    win.fill((30,30,30))
    cell_w = WIDTH // len(LETTERS)
    for i, ch in enumerate(LETTERS):
        color = (80,80,80)
        if i == idx: color = (0,150,255)
        pygame.draw.rect(win, color, (i*cell_w,0,cell_w,HEIGHT))
        text = font.render(ch, True, (255,255,255))
        win.blit(text, (i*cell_w+cell_w//2-text.get_width()//2, HEIGHT//2-text.get_height()//2))
    pygame.display.update()

while True:
    ret, frame = cap.read()
    if not ret: break

    pupil = detect_pupil(frame)
    if pupil:
        x, _ = pupil
        norm_x = x / frame.shape[1]
        smoothed_x = smoother.update(norm_x)
        idx = int(smoothed_x * len(LETTERS))
        idx = max(0, min(len(LETTERS)-1, idx))
    else:
        idx = prev_idx

    draw_keyboard(idx)

    # dwell selection
    if idx == prev_idx and idx is not None:
        if time.time() - last_time > DWELL_TIME:
            print("Selected:", LETTERS[idx])
            last_time = time.time()
    else:
        prev_idx = idx
        last_time = time.time()

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            exit()

    clock.tick(30)
