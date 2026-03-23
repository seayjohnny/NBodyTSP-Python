"""
Pygame-gui based UI Control Panel for N-Body TSP Simulator.

Renders to an offscreen pygame Surface which gets texture-blitted
over the OpenGL viewport by the renderer.

Mode-aware: different widgets for 2D vs torus modes.
"""

import pygame
import pygame_gui
from typing import Any, Optional

from tsp_nbody.themes import get_theme

PANEL_WIDTH = 280


class ControlPanel:
    """Interactive control panel using pygame_gui, rendered to an offscreen surface."""

    def __init__(self, panel_width: int = PANEL_WIDTH, mode: str = "torus",
                 theme_name: str = "warm", panel_height: int = 700):
        self.width = panel_width
        self.height = panel_height
        self.mode = mode
        self.theme_name = theme_name
        self.theme = get_theme(theme_name)

        # Offscreen surface for pygame-gui to render onto
        self.surface = pygame.Surface((panel_width, panel_height))

        # pygame-gui manager — renders to our offscreen surface
        self.manager = pygame_gui.UIManager(
            (panel_width, panel_height),
        )

        # Widget references
        self._widgets: dict[str, Any] = {}
        self._sliders: dict[str, pygame_gui.elements.UIHorizontalSlider] = {}
        self._slider_labels: dict[str, pygame_gui.elements.UILabel] = {}
        self._buttons: dict[str, pygame_gui.elements.UIButton] = {}
        self._dropdowns: dict[str, pygame_gui.elements.UIDropDownMenu] = {}

        # Stats labels (drawn manually, not via pygame-gui)
        self.stats: dict[str, str] = {}
        self.phase = "READY"
        self.fps = 0

        self._stats_y = 0
        self._font = None
        self._font_small = None

        self._build_widgets()

    def _add_slider(self, key: str, label: str, y: int, lo: float, hi: float, default: float,
                    fmt: str = ".2f") -> int:
        """Add a labeled slider. Returns the y position after the widget."""
        pad = 10
        w = self.width - 2 * pad
        display_val = int(default) if fmt == 'd' else default

        lbl = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(pad, y, w, 24),
            text=f"{label}: {display_val:{fmt}}",
            manager=self.manager,
        )
        self._slider_labels[key] = lbl
        self._widgets[f"{key}_label"] = lbl

        y += 24
        slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(pad, y, w, 24),
            start_value=default,
            value_range=(lo, hi),
            manager=self.manager,
        )
        self._sliders[key] = slider
        self._widgets[key] = slider

        # Store metadata
        slider._meta_key = key
        slider._meta_label = label
        slider._meta_fmt = fmt

        return y + 28

    def _add_button(self, key: str, label: str, x: int, y: int, w: int, h: int = 30) -> None:
        btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(x, y, w, h),
            text=label,
            manager=self.manager,
        )
        self._buttons[key] = btn

    def _add_dropdown(self, key: str, label: str, y: int, options: list[str], default: str) -> int:
        pad = 10
        w = self.width - 2 * pad

        lbl = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(pad, y, w, 24),
            text=label,
            manager=self.manager,
        )
        self._widgets[f"{key}_label"] = lbl
        y += 24

        dd = pygame_gui.elements.UIDropDownMenu(
            options_list=options,
            starting_option=default,
            relative_rect=pygame.Rect(pad, y, w, 30),
            manager=self.manager,
        )
        self._dropdowns[key] = dd
        return y + 36

    def _build_widgets(self):
        pad = 10
        w = self.width - 2 * pad
        y = 10

        # Title
        title_text = "TORUS CONTROLS" if self.mode == "torus" else "2D CONTROLS"
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(pad, y, w, 28),
            text=title_text,
            manager=self.manager,
        )
        y += 32

        if self.mode == "torus":
            y = self._build_torus_widgets(y)
        else:
            y = self._build_2d_widgets(y)

        # --- Common: MENU button ---
        y += 8
        self._add_button('menu', 'MENU (ESC)', pad, y, w, 28)
        y += 34

        # Stats start here
        self._stats_y = y

        # Fonts for stats drawing
        pygame.font.init()
        self._font = pygame.font.SysFont("consolas", 13)
        self._font_small = pygame.font.SysFont("consolas", 11)

    def _build_torus_widgets(self, y: int) -> int:
        y = self._add_slider('shrink_rate', 'Shrink Rate', y, 0.01, 0.50, 0.10)
        y = self._add_slider('epsilon', 'Epsilon', y, 0.01, 0.30, 0.08)
        y = self._add_slider('lj_strength', 'LJ Strength', y, 0.1, 5.0, 1.0)
        y = self._add_slider('perturbation', 'Perturbation', y, 0.0, 1.0, 0.50)
        y = self._add_slider('speed', 'Speed', y, 1, 64, 4, fmt='d')

        y += 6
        pad = 10
        w = self.width - 2 * pad
        btn_w = (w - 6) // 2
        self._add_button('collapse', 'COLLAPSE', pad, y, btn_w)
        self._add_button('reset', 'RESET', pad + btn_w + 6, y, btn_w)
        y += 36

        y = self._add_dropdown('embed_mode', 'Embedding', y, ['flat', 'centroid', 'linear'], 'flat')
        y = self._add_dropdown('ls_mode', 'Local Search', y, ['none', '2-opt', '3-opt', 'both'], 'none')

        return y

    def _build_2d_widgets(self, y: int) -> int:
        y = self._add_slider('wall_strength', 'Wall Strength', y, 100, 50000, 20000, fmt='d')
        y = self._add_slider('damp', 'Damping', y, 1.0, 100.0, 20.0, fmt='.1f')
        y = self._add_slider('dt', 'DT', y, 0.001, 0.05, 0.01, fmt='.3f')
        y = self._add_slider('dr', 'DR', y, 0.001, 0.05, 0.01, fmt='.3f')
        y = self._add_slider('slope_repulsion', 'Repulsion', y, 1.0, 200.0, 50.0, fmt='.1f')
        y = self._add_slider('mag_attraction', 'Attraction', y, 0.1, 100.0, 25.0, fmt='.1f')

        y += 6
        pad = 10
        w = self.width - 2 * pad
        btn_w = (w - 6) // 2
        self._add_button('start', 'START', pad, y, btn_w)
        self._add_button('reset', 'RESET', pad + btn_w + 6, y, btn_w)
        y += 36

        y = self._add_dropdown('wall_force', 'Wall Force Mode', y,
                               ['linear', 'inverse_square'], 'linear')

        return y

    def handle_event(self, event: pygame.event.Event, offset_x: int) -> dict[str, Any]:
        """
        Handle a pygame event. Events are translated so the panel is at x=0.
        Returns dict of changed values.
        """
        changes = {}

        # Translate mouse events so pygame-gui sees them relative to the panel
        translated = self._translate_event(event, offset_x)
        if translated is None:
            return changes

        self.manager.process_events(translated)

        # Check for button presses
        if translated.type == pygame_gui.UI_BUTTON_PRESSED:
            for key, btn in self._buttons.items():
                if translated.ui_element == btn:
                    changes[key] = True

        # Check for slider moves
        if translated.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            for key, slider in self._sliders.items():
                if translated.ui_element == slider:
                    val = slider.get_current_value()
                    fmt = getattr(slider, '_meta_fmt', '.2f')
                    label_text = getattr(slider, '_meta_label', key)
                    display_val = int(val) if fmt == 'd' else val
                    # Update label
                    if key in self._slider_labels:
                        self._slider_labels[key].set_text(f"{label_text}: {display_val:{fmt}}")
                    changes[key] = display_val

        # Check for dropdown changes
        if translated.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            for key, dd in self._dropdowns.items():
                if translated.ui_element == dd:
                    changes[key] = dd.selected_option[0]

        return changes

    def _translate_event(self, event: pygame.event.Event, offset_x: int) -> Optional[pygame.event.Event]:
        """Translate a mouse event so panel is at x=0. Returns None if outside panel."""
        if hasattr(event, 'pos'):
            mx, my = event.pos
            if mx < offset_x or mx >= offset_x + self.width:
                # Mouse outside panel — still pass non-positional events
                if event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
                    return None
            # Create a new event with translated position
            new_pos = (mx - offset_x, my)
            if event.type == pygame.MOUSEMOTION:
                return pygame.event.Event(event.type, pos=new_pos,
                                          rel=event.rel, buttons=event.buttons)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                return pygame.event.Event(event.type, pos=new_pos, button=event.button)
            elif event.type == pygame.MOUSEBUTTONUP:
                return pygame.event.Event(event.type, pos=new_pos, button=event.button)

        # Non-mouse events pass through as-is, including pygame_gui custom events
        return event

    def update(self, dt: float):
        """Update the pygame-gui manager. Call once per frame."""
        self.manager.update(dt)

    def set_slider_value(self, key: str, value: float):
        if key in self._sliders:
            self._sliders[key].set_current_value(value)
            fmt = getattr(self._sliders[key], '_meta_fmt', '.2f')
            label_text = getattr(self._sliders[key], '_meta_label', key)
            display_val = int(value) if fmt == 'd' else value
            if key in self._slider_labels:
                self._slider_labels[key].set_text(f"{label_text}: {display_val:{fmt}}")

    def set_button_disabled(self, key: str, disabled: bool):
        if key in self._buttons:
            if disabled:
                self._buttons[key].disable()
            else:
                self._buttons[key].enable()

    def update_stats(self, **kwargs):
        for k, v in kwargs.items():
            if k == 'phase':
                self.phase = str(v)
            elif k == 'fps':
                self.fps = int(v)
            else:
                self.stats[k] = str(v)

    def draw(self) -> pygame.Surface:
        """Draw the panel to the offscreen surface and return it."""
        theme = self.theme

        # Fill background
        self.surface.fill(theme["panel_bg"])

        # Let pygame-gui draw its widgets
        self.manager.draw_ui(self.surface)

        # Draw stats section manually below the widgets
        y = self._stats_y
        if self._font is None:
            return self.surface

        pad = 10

        # Separator line
        pygame.draw.line(self.surface, theme["panel_separator"],
                         (pad, y), (self.width - pad, y))
        y += 10

        # Phase
        phase_colors = {
            'TORUS': theme.get("panel_phase_torus", (46, 148, 80)),
            'READY': theme.get("panel_phase_torus", (46, 148, 80)),
            'COLLAPSING': theme.get("panel_phase_collapsing", (192, 88, 42)),
            'CIRCLE': theme.get("panel_phase_circle", (42, 120, 192)),
            'RUNNING': theme.get("panel_phase_collapsing", (192, 88, 42)),
            'FINISHED': theme.get("panel_phase_circle", (42, 120, 192)),
        }
        pc = phase_colors.get(self.phase, theme["panel_text"])
        surf = self._font.render(f"Phase: {self.phase}", True, pc)
        self.surface.blit(surf, (pad, y))
        y += 18

        # FPS
        surf = self._font_small.render(f"FPS: {self.fps}", True, theme["panel_text_dim"])
        self.surface.blit(surf, (pad, y))
        y += 16

        # Other stats
        for key, value in self.stats.items():
            nice = key.replace('_', ' ').title()
            lbl = self._font_small.render(f"{nice}:", True, theme["panel_text_dim"])
            val = self._font_small.render(value, True, theme["panel_stat_value"])
            self.surface.blit(lbl, (pad, y))
            self.surface.blit(val, (self.width - pad - val.get_width(), y))
            y += 15

        return self.surface
