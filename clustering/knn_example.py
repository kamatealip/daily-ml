import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pygame

Color = Tuple[int, int, int]

WIDTH = 1200
HEIGHT = 760
FPS = 60
CANVAS_RECT = pygame.Rect(40, 110, WIDTH - 80, HEIGHT - 140)
HEADER_RECT = pygame.Rect(28, 18, WIDTH - 56, 78)

BG_TOP = (10, 17, 27)
BG_BOTTOM = (3, 8, 15)
TEXT_PRIMARY = (234, 242, 255)
TEXT_MUTED = (144, 160, 183)
QUERY_COLOR = (244, 247, 255)

CLASS_COLORS: Sequence[Color] = (
    (255, 150, 122),
    (112, 222, 255),
    (160, 238, 166),
)

CLASS_NAMES: Sequence[str] = ("Class 1", "Class 2", "Class 3")


@dataclass
class SamplePoint:
    x: float
    y: float
    label: int
    color: Color
    pulse_seed: float


@dataclass
class Button:
    rect: pygame.Rect
    label: str
    color: Color


def clamp_channel(value: float) -> int:
    return max(0, min(255, int(value)))


def lerp(start: float, end: float, amount: float) -> float:
    return start + (end - start) * amount


def lerp_color(start: Color, end: Color, amount: float) -> Color:
    return (
        clamp_channel(lerp(start[0], end[0], amount)),
        clamp_channel(lerp(start[1], end[1], amount)),
        clamp_channel(lerp(start[2], end[2], amount)),
    )


def tint(color: Color, multiplier: float) -> Color:
    return (
        clamp_channel(color[0] * multiplier),
        clamp_channel(color[1] * multiplier),
        clamp_channel(color[2] * multiplier),
    )


def load_font(size: int, bold: bool = False) -> pygame.font.Font:
    for family in ("montserrat", "poppins", "segoeui", "verdana"):
        match = pygame.font.match_font(family, bold=bold)
        if match:
            return pygame.font.Font(match, size)
    return pygame.font.Font(None, size)


def make_background(width: int, height: int) -> pygame.Surface:
    background = pygame.Surface((width, height))
    for y in range(height):
        amount = y / max(1, height - 1)
        pygame.draw.line(
            background,
            lerp_color(BG_TOP, BG_BOTTOM, amount),
            (0, y),
            (width, y),
        )

    atmosphere = pygame.Surface((width, height), pygame.SRCALPHA)
    circles = (
        (180, 160, 250, (72, 110, 182, 60)),
        (950, 130, 290, (58, 95, 170, 56)),
        (780, 610, 340, (68, 122, 156, 36)),
        (260, 620, 300, (86, 66, 140, 24)),
    )
    for cx, cy, radius, color in circles:
        pygame.draw.circle(atmosphere, color, (cx, cy), radius)
    background.blit(atmosphere, (0, 0))
    return background


def make_grid_layer(width: int, height: int) -> pygame.Surface:
    grid = pygame.Surface((width, height), pygame.SRCALPHA)
    for x in range(0, width, 36):
        pygame.draw.line(grid, (118, 141, 173, 28), (x, 0), (x, height), 1)
    for y in range(0, height, 36):
        pygame.draw.line(grid, (118, 141, 173, 28), (0, y), (width, y), 1)
    return grid


def draw_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    color: Color,
    alpha: int,
    radius: int,
) -> None:
    panel = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(panel, (*color, alpha), panel.get_rect(), border_radius=radius)
    surface.blit(panel, rect.topleft)


def draw_button(
    surface: pygame.Surface,
    button: Button,
    font: pygame.font.Font,
    hovered: bool,
    enabled: bool = True,
) -> None:
    base_color = button.color if enabled else (72, 84, 102)
    fill = tint(base_color, 1.12 if hovered and enabled else 1.0)
    draw_panel(surface, button.rect, fill, 220 if enabled else 128, 14)
    pygame.draw.rect(surface, tint(fill, 1.3), button.rect, 2, border_radius=14)
    label = font.render(button.label, True, (12, 17, 29) if enabled else (170, 180, 198))
    surface.blit(label, label.get_rect(center=button.rect.center))


def add_sample(points: List[SamplePoint], x_pos: float, y_pos: float, label: int) -> None:
    points.append(
        SamplePoint(
            x=x_pos,
            y=y_pos,
            label=label,
            color=CLASS_COLORS[label],
            pulse_seed=random.uniform(0.0, math.tau),
        )
    )


def add_scatter(points: List[SamplePoint]) -> None:
    centers = (
        (CANVAS_RECT.left + 180, CANVAS_RECT.top + 160),
        (CANVAS_RECT.centerx + 10, CANVAS_RECT.bottom - 180),
        (CANVAS_RECT.right - 180, CANVAS_RECT.top + 190),
    )
    for label, (cx, cy) in enumerate(centers):
        for _ in range(16):
            angle = random.uniform(0.0, math.tau)
            distance = random.uniform(12.0, 95.0)
            x_pos = max(CANVAS_RECT.left + 16, min(CANVAS_RECT.right - 16, cx + math.cos(angle) * distance))
            y_pos = max(CANVAS_RECT.top + 16, min(CANVAS_RECT.bottom - 16, cy + math.sin(angle) * distance))
            add_sample(points, x_pos, y_pos, label)


def remove_nearest_sample(points: List[SamplePoint], x_pos: float, y_pos: float) -> bool:
    if not points:
        return False

    nearest_index = -1
    nearest_distance = float("inf")
    for index, point in enumerate(points):
        distance = math.hypot(point.x - x_pos, point.y - y_pos)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_index = index

    if nearest_distance <= 28.0:
        points.pop(nearest_index)
        return True
    return False


def predict_label(
    points: Sequence[SamplePoint],
    query_x: float,
    query_y: float,
    k_value: int,
) -> Tuple[Optional[int], List[SamplePoint]]:
    if not points:
        return None, []

    ranked_points = sorted(
        points,
        key=lambda point: (point.x - query_x) ** 2 + (point.y - query_y) ** 2,
    )
    neighbors = ranked_points[: max(1, min(k_value, len(ranked_points)))]

    counts = Counter(point.label for point in neighbors)
    predicted_label = max(
        counts,
        key=lambda label: (counts[label], -min(
            (point.x - query_x) ** 2 + (point.y - query_y) ** 2
            for point in neighbors
            if point.label == label
        )),
    )
    return predicted_label, neighbors


def draw_sample(surface: pygame.Surface, point: SamplePoint, runtime: float) -> None:
    center = (int(point.x), int(point.y))
    glow_radius = 10 + int(2 * math.sin(runtime * 4.2 + point.pulse_seed))
    glow = pygame.Surface((60, 60), pygame.SRCALPHA)
    pygame.draw.circle(glow, (*point.color, 78), (30, 30), glow_radius)
    surface.blit(glow, (center[0] - 30, center[1] - 30))
    pygame.draw.circle(surface, point.color, center, 8)
    pygame.draw.circle(surface, (245, 248, 255), center, 13, 2)


def draw_query(
    surface: pygame.Surface,
    x_pos: float,
    y_pos: float,
    predicted_label: Optional[int],
    runtime: float,
) -> None:
    color = CLASS_COLORS[predicted_label] if predicted_label is not None else QUERY_COLOR
    center = (int(x_pos), int(y_pos))
    pulse = 18 + int(3 * math.sin(runtime * 5.0))

    glow = pygame.Surface((140, 140), pygame.SRCALPHA)
    pygame.draw.circle(glow, (*color, 60), (70, 70), pulse + 8)
    pygame.draw.circle(glow, (*color, 120), (70, 70), pulse)
    surface.blit(glow, (center[0] - 70, center[1] - 70))

    pygame.draw.circle(surface, color, center, 11)
    pygame.draw.circle(surface, QUERY_COLOR, center, 18, 2)
    pygame.draw.line(surface, QUERY_COLOR, (center[0] - 20, center[1]), (center[0] + 20, center[1]), 2)
    pygame.draw.line(surface, QUERY_COLOR, (center[0], center[1] - 20), (center[0], center[1] + 20), 2)


def main() -> None:
    pygame.init()
    pygame.display.set_caption("KNN Playground")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    title_font = load_font(34, bold=True)
    body_font = load_font(22)
    small_font = load_font(18)

    background = make_background(WIDTH, HEIGHT)
    grid_layer = make_grid_layer(CANVAS_RECT.width, CANVAS_RECT.height)

    reset_button = Button(pygame.Rect(44, 32, 138, 50), "Reset", (255, 118, 128))
    scatter_button = Button(pygame.Rect(196, 32, 138, 50), "Scatter", (116, 173, 255))

    points: List[SamplePoint] = []
    active_label = 0
    k_value = 5
    runtime = 0.0
    status_text = "Left-click to place samples. Move inside the canvas to classify the query point."

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        runtime += dt
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    points.clear()
                    status_text = "Samples cleared. Add new labeled points."
                elif event.key == pygame.K_s:
                    points.clear()
                    add_scatter(points)
                    status_text = "Demo samples added. Move around the canvas to inspect KNN."
                elif event.key in (pygame.K_LEFTBRACKET, pygame.K_MINUS):
                    k_value = max(1, k_value - 1)
                    status_text = f"k set to {k_value}."
                elif event.key in (pygame.K_RIGHTBRACKET, pygame.K_EQUALS, pygame.K_PLUS):
                    k_value = min(15, k_value + 1)
                    status_text = f"k set to {k_value}."
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    active_label = event.key - pygame.K_1
                    status_text = f"Active label changed to {CLASS_NAMES[active_label]}."

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if reset_button.rect.collidepoint(event.pos):
                        points.clear()
                        status_text = "Samples cleared. Add new labeled points."
                    elif scatter_button.rect.collidepoint(event.pos):
                        points.clear()
                        add_scatter(points)
                        status_text = "Demo samples added. Move around the canvas to inspect KNN."
                    elif CANVAS_RECT.collidepoint(event.pos):
                        add_sample(points, float(event.pos[0]), float(event.pos[1]), active_label)
                        status_text = f"Added {CLASS_NAMES[active_label]} sample. Total samples: {len(points)}."
                elif event.button == 3 and CANVAS_RECT.collidepoint(event.pos):
                    removed = remove_nearest_sample(points, float(event.pos[0]), float(event.pos[1]))
                    status_text = "Nearest sample removed." if removed else "No nearby sample to remove."

        query_in_canvas = CANVAS_RECT.collidepoint(mouse_pos)
        predicted_label: Optional[int] = None
        neighbors: List[SamplePoint] = []
        if query_in_canvas:
            predicted_label, neighbors = predict_label(points, float(mouse_pos[0]), float(mouse_pos[1]), k_value)

        screen.blit(background, (0, 0))
        draw_panel(screen, HEADER_RECT, (15, 22, 36), 220, 24)
        draw_panel(screen, CANVAS_RECT, (10, 15, 26), 212, 22)
        pygame.draw.rect(screen, (86, 108, 142), CANVAS_RECT, 2, border_radius=22)
        screen.blit(grid_layer, CANVAS_RECT.topleft)

        line_layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        if query_in_canvas:
            for neighbor in neighbors:
                pygame.draw.line(
                    line_layer,
                    (*neighbor.color, 110),
                    (int(mouse_pos[0]), int(mouse_pos[1])),
                    (int(neighbor.x), int(neighbor.y)),
                    2,
                )
        screen.blit(line_layer, (0, 0))

        for point in points:
            draw_sample(screen, point, runtime)

        if query_in_canvas:
            draw_query(screen, float(mouse_pos[0]), float(mouse_pos[1]), predicted_label, runtime)

        draw_button(
            screen,
            reset_button,
            body_font,
            hovered=reset_button.rect.collidepoint(mouse_pos),
            enabled=bool(points),
        )
        draw_button(
            screen,
            scatter_button,
            body_font,
            hovered=scatter_button.rect.collidepoint(mouse_pos),
            enabled=True,
        )

        title_surface = title_font.render("K-Nearest Neighbors Playground", True, TEXT_PRIMARY)
        screen.blit(title_surface, (380, 28))

        hint_surface = small_font.render(
            "1 / 2 / 3 choose class   |   [ and ] change k   |   Right-click removes nearest sample",
            True,
            TEXT_MUTED,
        )
        screen.blit(hint_surface, (380, 63))

        active_k = min(k_value, len(points)) if points else k_value
        predicted_name = CLASS_NAMES[predicted_label] if predicted_label is not None else "None"
        stats_surface = small_font.render(
            f"Samples: {len(points)}    Active class: {CLASS_NAMES[active_label]}    k: {active_k}    Predicted: {predicted_name}",
            True,
            TEXT_MUTED if predicted_label is None else tint(CLASS_COLORS[predicted_label], 1.08),
        )
        screen.blit(stats_surface, (CANVAS_RECT.left + 18, CANVAS_RECT.top + 16))

        legend_x = CANVAS_RECT.right - 190
        legend_y = CANVAS_RECT.top + 18
        for index, class_name in enumerate(CLASS_NAMES):
            y_pos = legend_y + index * 30
            pygame.draw.circle(screen, CLASS_COLORS[index], (legend_x, y_pos), 8)
            label = f"{index + 1}: {class_name}"
            if index == active_label:
                label += " (active)"
            legend_surface = small_font.render(label, True, TEXT_PRIMARY if index == active_label else TEXT_MUTED)
            screen.blit(legend_surface, (legend_x + 18, y_pos - 11))

        status_surface = body_font.render(status_text, True, TEXT_PRIMARY)
        screen.blit(status_surface, (CANVAS_RECT.left + 18, CANVAS_RECT.bottom - 34))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
