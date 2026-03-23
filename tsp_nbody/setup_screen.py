"""
Setup Screen for N-Body TSP Simulator.

Uses pygame-menu for the launcher UI where users select dataset, mode,
wall force model, and color theme before starting the simulation.
"""

import pygame
import pygame_menu
from typing import Optional

from tsp_nbody.dataio import discover_datasets
from tsp_nbody.themes import THEMES, THEME_NAMES

SCREEN_W = 600
SCREEN_H = 500


def show_setup_screen() -> Optional[dict]:
    """
    Show the setup screen and return configuration, or None if user quits.

    Returns:
        dict with keys: dataset_path, dataset_name, mode, wall_force_mode, theme
    """
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("N-Body TSP Simulator")

    # Discover datasets
    datasets = discover_datasets()
    if not datasets:
        datasets = [{"name": "grid4x4", "path": "datasets/grid4x4/coords.txt", "n_cities": 16, "has_optimal": True}]

    # State that will be captured by callbacks
    state = {
        "dataset_idx": 0,
        "mode": "torus",
        "wall_force_mode": "linear",
        "theme": "warm",
        "started": False,
        "quit": False,
    }

    # --- Custom theme for the menu ---
    menu_theme = pygame_menu.Theme(
        background_color=(244, 241, 236),
        title_background_color=(192, 88, 42),
        title_font_color=(255, 255, 255),
        title_font=pygame_menu.font.FONT_OPEN_SANS_BOLD,
        title_font_size=28,
        widget_font=pygame_menu.font.FONT_OPEN_SANS,
        widget_font_size=18,
        widget_font_color=(42, 37, 32),
        selection_color=(192, 88, 42),
        widget_margin=(0, 8),
        widget_padding=(4, 12),
    )

    menu = pygame_menu.Menu(
        "N-Body TSP Simulator",
        SCREEN_W, SCREEN_H,
        theme=menu_theme,
    )

    # --- Dataset selector ---
    dataset_items = [(f"{d['name']} ({d['n_cities']} cities)", i) for i, d in enumerate(datasets)]

    def on_dataset_change(value, idx):
        state["dataset_idx"] = idx

    menu.add.dropselect(
        "Dataset  ",
        dataset_items,
        default=0,
        onchange=on_dataset_change,
        selection_box_width=280,
        placeholder_add_to_selection_box=False,
    )

    # --- Mode selector ---
    mode_items = [("Torus 3D", "torus"), ("2D Walls", "2d")]

    def on_mode_change(value, mode_val):
        state["mode"] = mode_val
        # Update wall force widget state
        if wall_force_widget is not None:
            if mode_val == "torus":
                wall_force_widget.hide()
                wall_force_label.show()
            else:
                wall_force_widget.show()
                wall_force_label.hide()

    menu.add.dropselect(
        "Mode  ",
        mode_items,
        default=0,
        onchange=on_mode_change,
    )

    # --- Wall force model ---
    wall_items = [("Linear Spring", "linear"), ("Inverse Square", "inverse_square")]

    def on_wall_change(value, wf_val):
        state["wall_force_mode"] = wf_val

    wall_force_widget = menu.add.dropselect(
        "Wall Force  ",
        wall_items,
        default=0,
        onchange=on_wall_change,
    )
    # Hidden by default since torus is default mode
    wall_force_widget.hide()

    wall_force_label = menu.add.label(
        "Wall Force: Inverse Square (fixed for Torus)",
        font_size=14,
        font_color=(138, 125, 110),
    )

    # --- Theme selector ---
    theme_items = [(THEMES[name]["name"], name) for name in THEME_NAMES]

    def on_theme_change(value, theme_val):
        state["theme"] = theme_val

    menu.add.dropselect(
        "Theme  ",
        theme_items,
        default=THEME_NAMES.index("warm"),
        onchange=on_theme_change,
    )

    menu.add.vertical_margin(20)

    # --- Start button ---
    def on_start():
        state["started"] = True

    menu.add.button(
        "Start Simulation",
        on_start,
        font_color=(255, 255, 255),
        background_color=(192, 88, 42),
        selection_color=(255, 255, 255),
        font_size=20,
        padding=(10, 30),
        border_width=0,
        cursor=pygame_menu.locals.CURSOR_HAND,
    )

    # --- Main loop ---
    clock = pygame.time.Clock()
    running = True

    while running:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                state["quit"] = True
                running = False
                break
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                state["quit"] = True
                running = False
                break

        if not running:
            break

        if state["started"]:
            running = False
            break

        if menu.is_enabled():
            menu.update(events)
            menu.draw(screen)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

    if state["quit"]:
        return None

    ds = datasets[state["dataset_idx"]]
    return {
        "dataset_path": ds["path"],
        "dataset_name": ds["name"],
        "mode": state["mode"],
        "wall_force_mode": state["wall_force_mode"],
        "theme": state["theme"],
    }
