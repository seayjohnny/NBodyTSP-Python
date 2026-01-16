# ruff:noqa: F403 F405
"""
OpenGL Renderer for N-Body TSP Simulator

Provides real-time GPU-accelerated visualization using OpenGL.
"""

import numpy as np
from typing import Optional, Tuple, TypedDict, TypeAlias
import sys

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    import pygame
    from pygame.locals import *
    OPENGL_AVAILABLE = True
except ImportError:
    print("Warning: OpenGL/Pygame not available. Rendering disabled.")
    OPENGL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("Warning: OpenCV not available. Video recording disabled.")
    CV2_AVAILABLE = False

try:
    import win32gui
    WIN_GUI_AVAILABLE = True
except ImportError:
    WIN_GUI_AVAILABLE = False


ColorType: TypeAlias = Tuple[float, float, float]

class RendererOptions(TypedDict, total=False):
    window_size: Tuple[int, int]
    title: str
    city_size: float
    path_width: float
    wall_width: float
    padding: float  # Padding as fraction of view (e.g., 0.1 = 10%)
    show_grid: bool
    show_density: bool
    show_bubbles: bool
    use_random_city_colors: bool
    color_background: ColorType
    color_city: ColorType
    color_path: ColorType
    color_wall_contract: ColorType
    color_wall_static: ColorType
    color_wall_expand: ColorType
    color_bubble: ColorType
    color_density: ColorType
    color_text: ColorType
    record_video: bool
    video_fps: int


default_renderer_options: RendererOptions = {
    "window_size": (800, 800),
    "title": "N-Body TSP Simulator",
    "city_size": 5.0,
    "path_width": 2.0,
    "wall_width": 2.0,
    "padding": 0.1,  # 10% padding around edges
    "show_grid": False,
    "show_density": False,
    "show_bubbles": False,
    "use_random_city_colors": False,
    "color_background": (1.0,1.0, 1.0),
    "color_city": (0.2, 0.0, 1.0),
    "color_path": (0.0, 0.5, 0.0),
    "color_wall_contract": (1.0, 0.0, 0.0),
    "color_wall_static": (0.3, 0.3, 0.0),
    "color_wall_expand": (0.0, 0.0, 1.0),
    "color_bubble": (0.2, 0.8, 1.0),
    "color_density": (0.5, 0.5, 0.5),
    "color_text": (1.0, 1.0, 1.0),
    "record_video": False,
    "video_fps": 30,
}


class TSPRenderer:
    """OpenGL renderer for visualizing N-body TSP simulation."""
    
    def __init__(self, options: RendererOptions = default_renderer_options):
        """
        Initialize the renderer.

        Args:
            window_size: (width, height) of window
            title: Window title
        """

        self.window_size = options.get("window_size", (800, 800))
        self.title = options.get("title", "N-Body TSP Simulator")
        self.window = None
        self.clock = None
        self.font = None
        self.is_initialized = False
        self.is_paused = False
        self.bubble_placement_mode = False
        self.manual_bubbles = []  # List of (x, y) positions in normalized coordinates

        # Sizes
        self.city_size = options.get("city_size", 5.0)  # Point size for cities
        self.path_width = options.get("path_width", 2.0)  # Line width for paths
        self.wall_width = options.get("wall_width", 2.0)  # Line width for walls
        self.padding = options.get("padding", 0.1)  # Padding fraction

        # Drawing options
        self.show_grid = options.get("show_grid", False)
        self.show_density = options.get("show_density", False)
        self.show_bubbles = options.get("show_bubbles", False)
        self.use_random_city_colors = options.get("use_random_city_colors", False)

        # Colors (RGB)
        self.color_background = options.get("color_background", (0.0, 0.0, 0.0))
        self.color_city = options.get("color_city", (1.0, 0.0, 0.0))
        self.color_path = options.get("color_path", (0.0, 1.0, 0.0))
        self.color_wall_contract = options.get("color_wall_contract", (1.0, 0.0, 0.0))
        self.color_wall_static = options.get("color_wall_static", (1.0, 1.0, 0.0))
        self.color_wall_expand = options.get("color_wall_expand", (0.0, 0.0, 1.0))
        self.color_bubble = options.get("color_bubble", (0.2, 0.8, 1.0))
        self.color_density = options.get("color_density", (0.5, 0.5, 0.5))
        self.color_text = options.get("color_text", (1.0, 1.0, 1.0))

        # Per-city colors (will be set when number of cities is known)
        self.city_colors = None

        if not OPENGL_AVAILABLE:
            print("Renderer created but OpenGL not available")
    
    def initialize(self) -> bool:
        """
        Initialize OpenGL window and context.
        
        Returns:
            True if successful, False otherwise
        """
        if not OPENGL_AVAILABLE:
            print("Cannot initialize: OpenGL not available")
            return False
        
        try:
            # Initialize Pygame
            pygame.init()
            
            # Set up display with OpenGL
            self.window = pygame.display.set_mode(self.window_size, DOUBLEBUF | OPENGL)
            pygame.display.set_caption(self.title)
            
            # Initialize clock for frame timing
            self.clock = pygame.time.Clock()
            
            # Set up OpenGL viewport
            glViewport(0, 0, self.window_size[0], self.window_size[1])

            # Set up 2D orthographic projection with padding
            self.setup_padded_projection()
            # Enable features
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_POINT_SMOOTH)
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
            
            # Set background color
            glClearColor(*self.color_background, 1.0)

            # Initialize font for text rendering
            try:
                self.font = pygame.font.Font(None, 24)  # Default system font, size 24
            except Exception as e:
                print(f"Warning: Failed to initialize font: {e}")
                self.font = None

            # Focus window on Windows
            if WIN_GUI_AVAILABLE:
                self.focus_window()

            self.is_initialized = True
            print(f"Renderer initialized: {self.window_size[0]}x{self.window_size[1]}")
            return True

        except Exception as e:
            print(f"Failed to initialize renderer: {e}")
            return False

    def setup_padded_projection(self):
        """
        Set up the orthographic projection with padding.

        Since the simulator normalizes everything so the outer wall is at 1.0,
        we just need a fixed view with padding around it.
        """
        # View limit with padding (outer wall normalized to 1.0)
        view_limit = 1.0 + self.padding

        # Update projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-view_limit, view_limit, -view_limit, view_limit, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def generate_random_colors(self, n_cities: int):
        """
        Generate random colors for each city.

        Args:
            n_cities: Number of cities (points) to generate colors for
        """
        # Generate random RGB colors for each city
        # Use HSV color space for better color distribution
        np.random.seed(42)  # Use a fixed seed for consistency

        # Generate colors in HSV space and convert to RGB
        colors = []
        for i in range(n_cities):
            hue = (i / n_cities) * 360  # Distribute hues evenly
            saturation = 0.7 + np.random.rand() * 0.3  # High saturation (70-100%)
            value = 0.5 + np.random.rand() * 0.3  # High value (70-100%)

            # Convert HSV to RGB
            c = value * saturation
            x = c * (1 - abs((hue / 60) % 2 - 1))
            m = value - c

            if hue < 60:
                r, g, b = c, x, 0
            elif hue < 120:
                r, g, b = x, c, 0
            elif hue < 180:
                r, g, b = 0, c, x
            elif hue < 240:
                r, g, b = 0, x, c
            elif hue < 300:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x

            colors.append((r + m, g + m, b + m))

        self.city_colors = np.array(colors, dtype=np.float32)
        print(f"Generated {n_cities} random colors for cities")

    def clear(self):
        """Clear the display."""
        if not self.is_initialized:
            return
        glClear(GL_COLOR_BUFFER_BIT)
    
    def draw_cities(self, positions: np.ndarray, size: float | None = None,
                   color: Optional[Tuple[float, float, float]] = None):
        """
        Draw cities as points.

        Args:
            positions: City positions array (n, 2)
            size: Point size
            color: RGB color tuple, or None for default (uses per-city colors if available)
        """
        if not self.is_initialized:
            return

        # Generate random colors if needed and not already generated
        if self.use_random_city_colors and self.city_colors is None:
            self.generate_random_colors(len(positions))

        _size = size if size is not None else self.city_size

        
        black = (0.0, 0.0, 0.0)

        # Draw outline
        glColor3f(*black)
        glPointSize(_size + 2)
        glBegin(GL_POINTS)
        for pos in positions:
            glVertex2f(pos[0], pos[1])
        glEnd()

        glPointSize(_size)
        # Use per-city colors if available and no override color specified
        if self.city_colors is not None and self.use_random_city_colors:
            glBegin(GL_POINTS)
            for i, pos in enumerate(positions):
                # Draw city point
                if i < len(self.city_colors):
                    glColor3f(*self.city_colors[i])
                else:
                    glColor3f(*self.color_city)
                glVertex2f(pos[0], pos[1])
            glEnd()
        else:
            # Use single color for all points
            color = color or self.color_city
            glColor3f(*color)
            glPointSize(_size)
            glBegin(GL_POINTS)
            for pos in positions:
                # Draw city point
                glVertex2f(pos[0], pos[1])
            glEnd()
    
    def draw_circle(self, center: Tuple[float, float], radius: float, 
                   segments: int = 100, color: Optional[Tuple[float, float, float]] = None,
                   width: float = 1.0, filled: bool = False):
        """
        Draw a circle.
        
        Args:
            center: (x, y) center position
            radius: Circle radius
            segments: Number of line segments
            color: RGB color tuple
            width: Line width
            filled: Whether to fill the circle
        """
        if not self.is_initialized:
            return
        
        color = color or (1.0, 1.0, 1.0)
        
        glLineWidth(width)
        glColor3f(*color)
        
        mode = GL_POLYGON if filled else GL_LINE_LOOP
        
        glBegin(mode)
        for i in range(segments):
            theta = 2.0 * np.pi * i / segments
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            glVertex2f(x, y)
        glEnd()
    
    def draw_walls(self, inner_radius: float, outer_radius: float,
                  inner_direction: int, outer_direction: int,
                  width: float | None = None):
        """
        Draw inner and outer circular walls with color-coded directions.
        
        Args:
            inner_radius: Inner wall radius
            outer_radius: Outer wall radius
            inner_direction: -1 (contract), 0 (static), 1 (expand)
            outer_direction: -1 (contract), 0 (static), 1 (expand)
        """
        if not self.is_initialized:
            return
        
        _width = width if width is not None else self.wall_width

        # Color code based on direction
        def get_color(direction):
            if direction < 0:
                return self.color_wall_contract  # Red: contracting
            elif direction == 0:
                return self.color_wall_static    # Yellow: static
            else:
                return self.color_wall_expand    # Blue: expanding
        
        # Draw inner wall
        if inner_radius > 0.001:
            inner_color = get_color(inner_direction)
            self.draw_circle((0, 0), inner_radius, color=inner_color, width=_width)
        
        # Draw outer wall
        outer_color = get_color(outer_direction)
        self.draw_circle((0, 0), outer_radius, color=outer_color, width=width)
    
    def draw_path(self, coords: np.ndarray, path: np.ndarray,
                 width: float | None = None, color: Optional[Tuple[float, float, float]] = None):
        """
        Draw TSP path connecting cities.
        
        Args:
            coords: City coordinates (n, 2)
            path: Array of city indices in visit order
            width: Line width
            color: RGB color tuple
        """
        if not self.is_initialized or len(path) == 0:
            return
        
        color = color or self.color_path
        _width = width if width is not None else self.path_width

        glLineWidth(_width)
        glColor3f(*color)
        
        glBegin(GL_LINE_LOOP)
        for idx in path:
            glVertex2f(coords[idx, 0], coords[idx, 1])
        glEnd()
    
    def draw_bubbles(self, bubbles: np.ndarray, alpha: float = 0.3):
        """
        Draw density bubbles.

        Args:
            bubbles: Array of shape (n, 4) with (x, y, radius, active)
            alpha: Transparency (0-1)
        """
        if not self.is_initialized or not self.show_bubbles or bubbles is None or len(bubbles) == 0:
            return

        for bubble in bubbles:
            if bubble[3] > 0.5:  # Check if active
                x, y, r = bubble[0], bubble[1], bubble[2]
                # Draw filled circle with transparency
                glColor4f(*self.color_bubble, alpha)
                glBegin(GL_POLYGON)
                for i in range(50):
                    theta = 2.0 * np.pi * i / 50
                    bx = x + r * np.cos(theta)
                    by = y + r * np.sin(theta)
                    glVertex2f(bx, by)
                glEnd()

                # Draw outline (opaque)
                glColor3f(*self.color_bubble)
                glLineWidth(2.0)
                glBegin(GL_LINE_LOOP)
                for i in range(50):
                    theta = 2.0 * np.pi * i / 50
                    bx = x + r * np.cos(theta)
                    by = y + r * np.sin(theta)
                    glVertex2f(bx, by)
                glEnd()
    
    def draw_density_grid(self, density: Optional[np.ndarray],
                         bins: int, max_density: Optional[float] = None):
        """
        Draw density heatmap as colored rectangles with grid lines.

        Args:
            density: Density values (bins*bins,), or None
            bins: Number of bins per dimension
            max_density: Maximum density for color scaling
            draw_grid_lines: Whether to draw grid lines
        """
        if not self.is_initialized or (not self.show_density and not self.show_grid):
            return

        min_x, min_y = -1.0, -1.0
        cell_width = 2.0 / bins
        cell_height = 2.0 / bins

        # Draw colored rectangles for density (if data provided)
        if density is not None and len(density) > 0 and self.show_density:
            if max_density is None:
                max_density = np.max(density) if len(density) > 0 else 1.0
                # Ensure we don't divide by zero
                if max_density == 0:
                    max_density = 1.0

            for i in range(bins):
                for j in range(bins):
                    idx = i + j * bins
                    if idx >= len(density):
                        continue

                    d = density[idx]
                    if d > 0:
                        # Color intensity based on density (like CUDA: density/sqrt(N))
                        # Using sqrt scaling to make differences more visible
                        intensity = min(np.sqrt(d / max_density), 1.0)

                        # Use a color gradient: we use color density as base
                        # More intense = more particles
                        r = self.color_density[0] * intensity
                        g = self.color_density[1] * intensity
                        b = self.color_density[2] * intensity
                        glColor4f(r, g, b, 0.6)

                        # Draw filled rectangle
                        # NOTE: Grid y=0 is at the TOP, so we flip when drawing
                        x = min_x + i * cell_width
                        y = min_y + (bins - 1 - j) * cell_height

                        glBegin(GL_QUADS)
                        glVertex2f(x, y)
                        glVertex2f(x + cell_width, y)
                        glVertex2f(x + cell_width, y + cell_height)
                        glVertex2f(x, y + cell_height)
                        glEnd()

        # Always draw grid lines
        if self.show_grid:
            glLineWidth(1.0)
            glColor4f(0.3, 0.3, 0.3, 0.5)  # Semi-transparent gray

            # Vertical lines
            for i in range(bins + 1):
                x = min_x + i * cell_width
                glBegin(GL_LINES)
                glVertex2f(x, min_y)
                glVertex2f(x, min_y + bins * cell_height)
                glEnd()

            # Horizontal lines
            for j in range(bins + 1):
                y = min_y + j * cell_height
                glBegin(GL_LINES)
                glVertex2f(min_x, y)
                glVertex2f(min_x + bins * cell_width, y)
                glEnd()

    def draw_text(self, text: str, position: Tuple[int, int],
                  color: Optional[Tuple[int, int, int]] = None):
        """
        Draw text overlay on the screen.

        Args:
            text: Text to display
            position: (x, y) position in screen coordinates (pixels from top-left)
            color: RGB color tuple (0-255 range), or None for default white
        """
        return
        if not self.is_initialized or self.font is None:
            return

        color = color or (255, 255, 255)  # White by default

        # Render text to a surface
        text_surface = self.font.render(text, True, color)

        # Get text dimensions
        text_width = text_surface.get_width()
        text_height = text_surface.get_height()

        # Convert surface to raw data
        text_data = pygame.image.tostring(text_surface, "RGBA", True)

        # Save OpenGL state
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()

        # Set up screen-space orthographic projection
        glOrtho(0, self.window_size[0], self.window_size[1], 0, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # Position raster
        glRasterPos2i(position[0], position[1])

        # Draw pixels
        glDrawPixels(text_width, text_height, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        # Restore OpenGL state
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopAttrib()

    def draw_pressure_overlay(self, pressure: float, wall_gap: float):
        """
        Draw wall pressure information as text overlay.

        Args:
            pressure: Current wall pressure value
            wall_gap: Distance between inner and outer walls
        """
        if not self.is_initialized or self.font is None:
            return

        # Format text with pressure info
        pressure_text = f"Wall Pressure: {pressure:.2f}"
        gap_text = f"Wall Gap: {wall_gap:.3f}"

        # Position in top-left corner with padding
        padding = 20
        line_height = 25

        # Draw pressure with color based on magnitude
        if pressure > 100:
            color = (255, 0, 0)  # Red for high pressure
        elif pressure > 50:
            color = (255, 255, 0)  # Yellow for medium pressure
        else:
            color = (255, 255, 255)  # White for low pressure

        self.draw_text(pressure_text, (padding, padding), color)
        self.draw_text(gap_text, (padding, padding + line_height), (255, 255, 255))

    def draw_pause_indicator(self):
        """Draw 'PAUSED' text overlay when simulation is paused."""
        if not self.is_initialized or self.font is None or not self.is_paused:
            return

        # Draw "PAUSED" in the center of the screen
        pause_text = "PAUSED (Click to Resume)"
        text_surface = self.font.render(pause_text, True, (255, 255, 0))  # Yellow

        # Calculate center position
        text_width = text_surface.get_width()
        x = (self.window_size[0] - text_width) // 2
        y = self.window_size[1] // 2 - 50

        # Draw with a semi-transparent black background for visibility
        bg_padding = 10
        bg_width = text_width + 2 * bg_padding
        bg_height = text_surface.get_height() + 2 * bg_padding

        # Save state
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.window_size[0], self.window_size[1], 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # Draw semi-transparent background
        glColor4f(0.0, 0.0, 0.0, 0.7)
        glBegin(GL_QUADS)
        glVertex2f(x - bg_padding, y - bg_padding)
        glVertex2f(x + text_width + bg_padding, y - bg_padding)
        glVertex2f(x + text_width + bg_padding, y + text_surface.get_height() + bg_padding)
        glVertex2f(x - bg_padding, y + text_surface.get_height() + bg_padding)
        glEnd()

        # Draw text
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        glRasterPos2i(x, y)
        glDrawPixels(text_width, text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        # Restore state
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopAttrib()

    def draw_bubble_placement_indicator(self):
        """Draw bubble placement mode instructions."""
        if not self.is_initialized or self.font is None or not self.bubble_placement_mode:
            return

        # Draw instructions
        instructions = [
            "BUBBLE PLACEMENT MODE",
            "Left Click: Place Bubble",
            "Right Click: Remove Bubble",
            f"Bubbles: {len(self.manual_bubbles)}",
            "",
            "SPACE: Start with manual bubbles",
            "D: Use dynamic bubbles instead"
        ]

        padding = 20
        line_height = 25

        for i, text in enumerate(instructions):
            if text == "":
                continue
            color = (0, 255, 255) if i == 0 else (255, 255, 255)  # Cyan for title
            self.draw_text(text, (padding, padding + i * line_height), color)

    def draw_manual_bubbles(self, initial_radius: float = 0.15):
        """
        Draw manually placed bubbles during placement mode.

        Args:
            initial_radius: Initial radius for bubbles
        """
        if not self.is_initialized or not self.bubble_placement_mode:
            return

        for bx, by in self.manual_bubbles:
            # Draw filled circle with transparency
            glColor4f(*self.color_bubble, 0.3)
            glBegin(GL_POLYGON)
            for i in range(50):
                theta = 2.0 * np.pi * i / 50
                x = bx + initial_radius * np.cos(theta)
                y = by + initial_radius * np.sin(theta)
                glVertex2f(x, y)
            glEnd()

            # Draw outline (opaque)
            glColor3f(*self.color_bubble)
            glLineWidth(2.0)
            glBegin(GL_LINE_LOOP)
            for i in range(50):
                theta = 2.0 * np.pi * i / 50
                x = bx + initial_radius * np.cos(theta)
                y = by + initial_radius * np.sin(theta)
                glVertex2f(x, y)
            glEnd()
    
    def draw_frame_complete(self, positions: np.ndarray, inner_radius: float,
                           outer_radius: float, inner_dir: int, outer_dir: int,
                           bubbles: Optional[np.ndarray] = None,
                           density_grid: Optional[np.ndarray] = None,
                           grid_bins: int = 8,
                           path: Optional[np.ndarray] = None,
                           coords: Optional[np.ndarray] = None,
                           pressure: Optional[float] = None):
        """
        Draw a complete frame with all elements.

        Args:
            positions: Current city positions
            inner_radius: Inner wall radius
            outer_radius: Outer wall radius
            inner_dir: Inner wall direction
            outer_dir: Outer wall direction
            bubbles: Optional bubble array
            density_grid: Optional density grid array
            grid_bins: Number of grid bins per dimension
            path: Optional path to draw
            coords: Original coordinates (needed if drawing path)
            pressure: Optional wall pressure to display
        """
        self.clear()

        # Draw density grid (always draw grid lines, color cells if data available)
        self.draw_density_grid(density_grid, grid_bins)

        # Draw walls
        self.draw_walls(inner_radius, outer_radius, inner_dir, outer_dir, width=self.wall_width)
        if bubbles is not None:
            self.draw_bubbles(bubbles)

        # Draw path if provided
        if path is not None and coords is not None:
            self.draw_path(coords, path, width=self.path_width, color=self.color_path)

        # Draw cities (foreground)
        self.draw_cities(positions, size=self.city_size, color=self.color_city)

        # Draw pressure overlay (foreground)
        if pressure is not None:
            wall_gap = outer_radius - inner_radius
            self.draw_pressure_overlay(pressure, wall_gap)

        # Draw pause indicator if paused
        self.draw_pause_indicator()

        self.swap_buffers()
    
    def swap_buffers(self):
        """Display the rendered frame."""
        if not self.is_initialized:
            return
        pygame.display.flip()

    def capture_frame(self, from_back_buffer: bool = False) -> Optional[np.ndarray]:
        """
        Capture the current framebuffer as a numpy array.

        Args:
            from_back_buffer: If True, read from back buffer (before swap),
                            if False, read from front buffer (after swap)

        Returns:
            RGB image array of shape (height, width, 3) or None if capture fails
        """
        if not self.is_initialized:
            return None

        try:
            # Select which buffer to read from
            if from_back_buffer:
                glReadBuffer(GL_BACK)
            else:
                glReadBuffer(GL_FRONT)

            # Read pixels from the selected buffer
            width, height = self.window_size
            glPixelStorei(GL_PACK_ALIGNMENT, 1)
            data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)

            # Convert to numpy array
            image = np.frombuffer(data, dtype=np.uint8)
            image = image.reshape(height, width, 3)

            # Flip vertically (OpenGL has origin at bottom-left, images at top-left)
            image = np.flipud(image)

            return image
        except Exception as e:
            print(f"Warning: Failed to capture frame: {e}")
            return None
    
    def handle_events(self) -> bool:
        """
        Process window events.

        Returns:
            True if should continue, False if quit requested
        """
        if not self.is_initialized:
            return False

        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return False
            if event.type == MOUSEBUTTONDOWN:
                if self.bubble_placement_mode:
                    # Bubble placement mode: left click adds, right click removes
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    world_x, world_y = self.screen_to_world(mouse_x, mouse_y)

                    if event.button == 1:  # Left mouse button - add bubble
                        self.manual_bubbles.append((world_x, world_y))
                        print(f"Bubble placed at ({world_x:.2f}, {world_y:.2f}). Total: {len(self.manual_bubbles)}")
                    elif event.button == 3:  # Right mouse button - remove nearest bubble
                        if self.manual_bubbles:
                            # Find and remove nearest bubble
                            min_dist = float('inf')
                            nearest_idx = -1
                            for i, (bx, by) in enumerate(self.manual_bubbles):
                                dist = np.sqrt((bx - world_x)**2 + (by - world_y)**2)
                                if dist < min_dist:
                                    min_dist = dist
                                    nearest_idx = i

                            if nearest_idx >= 0 and min_dist < 0.2:  # Only remove if close enough
                                removed = self.manual_bubbles.pop(nearest_idx)
                                print(f"Bubble removed at ({removed[0]:.2f}, {removed[1]:.2f}). Total: {len(self.manual_bubbles)}")
                            else:
                                print("No bubble close enough to remove")
                else:
                    # Normal mode: left click toggles pause
                    if event.button == 1:  # Left mouse button
                        self.toggle_pause()

        return True

    def toggle_pause(self):
        """Toggle pause state and print status."""
        self.is_paused = not self.is_paused
        status = "PAUSED" if self.is_paused else "RESUMED"
        print(f"Simulation {status}")

    def screen_to_world(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """
        Convert screen coordinates to world coordinates.

        Args:
            screen_x: X coordinate in screen space (pixels from left)
            screen_y: Y coordinate in screen space (pixels from top)

        Returns:
            Tuple of (world_x, world_y) in normalized simulation space [-1, 1]
        """
        # Convert screen coordinates to normalized device coordinates
        # Screen origin is top-left, OpenGL origin is bottom-left
        ndc_x = (2.0 * screen_x / self.window_size[0]) - 1.0
        ndc_y = 1.0 - (2.0 * screen_y / self.window_size[1])  # Flip Y axis

        # Account for padding in the projection
        view_limit = 1.0 + self.padding
        world_x = ndc_x * view_limit
        world_y = ndc_y * view_limit

        return (world_x, world_y)
    
    def wait_for_key(self):
        """Wait for user to press a key."""
        if not self.is_initialized:
            return

        if self.bubble_placement_mode:
            print("Press any key to start simulation...")
        else:
            print("Press any key to continue...")

        waiting = True
        while waiting:
            # Process events to handle mouse clicks during placement mode
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN:
                    # Exit bubble placement mode when key is pressed
                    if self.bubble_placement_mode:
                        self.bubble_placement_mode = False
                        print(f"Starting simulation with {len(self.manual_bubbles)} manual bubbles")
                    waiting = False
                if event.type == MOUSEBUTTONDOWN and self.bubble_placement_mode:
                    # Handle bubble placement/removal during wait
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    world_x, world_y = self.screen_to_world(mouse_x, mouse_y)

                    if event.button == 1:  # Left mouse button - add bubble
                        self.manual_bubbles.append((world_x, world_y))
                        print(f"Bubble placed at ({world_x:.2f}, {world_y:.2f}). Total: {len(self.manual_bubbles)}")
                    elif event.button == 3:  # Right mouse button - remove nearest bubble
                        if self.manual_bubbles:
                            # Find and remove nearest bubble
                            min_dist = float('inf')
                            nearest_idx = -1
                            for i, (bx, by) in enumerate(self.manual_bubbles):
                                dist = np.sqrt((bx - world_x)**2 + (by - world_y)**2)
                                if dist < min_dist:
                                    min_dist = dist
                                    nearest_idx = i

                            if nearest_idx >= 0 and min_dist < 0.2:  # Only remove if close enough
                                removed = self.manual_bubbles.pop(nearest_idx)
                                print(f"Bubble removed at ({removed[0]:.2f}, {removed[1]:.2f}). Total: {len(self.manual_bubbles)}")
                            else:
                                print("No bubble close enough to remove")

            pygame.time.wait(10)

    def wait_for_unpause(self):
        """Wait until simulation is unpaused."""
        if not self.is_initialized:
            return

        if not self.is_paused:
            return

        while self.is_paused:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        self.toggle_pause()
            pygame.time.wait(10)
    
    def limit_framerate(self, fps: int = 60):
        """
        Limit rendering frame rate.
        
        Args:
            fps: Target frames per second
        """
        if self.clock:
            self.clock.tick(fps)
    
    def focus_window(self):
        """Bring the window to the foreground (Windows only)."""
        if not self.is_initialized or not WIN_GUI_AVAILABLE:
            return
        
        try:
            hwnd = pygame.display.get_wm_info()['window']
            win32gui.SetForegroundWindow(hwnd)
            print("Window focused")
        except Exception as e:
            print(f"Failed to focus window: {e}")

    def close(self):
        """Clean up and close the window."""
        if self.is_initialized:
            pygame.quit()
            self.is_initialized = False
            print("Renderer closed")


# Example usage
if __name__ == "__main__":
    # Create test data
    np.random.seed(42)
    n_cities = 30
    
    # Random cities in a circle
    angles = np.linspace(0, 2*np.pi, n_cities, endpoint=False)
    positions = np.column_stack([
        0.8 * np.cos(angles),
        0.8 * np.sin(angles)
    ])
    
    # Initialize renderer
    renderer = TSPRenderer()
    if not renderer.initialize():
        print("Failed to initialize renderer")
        sys.exit(1)
    
    # Animation loop
    print("Running animation test...")
    print("Press ESC to quit")
    
    inner_radius = 0.0
    outer_radius = 1.0
    frame = 0
    
    running = True
    while running and frame < 300:
        # Handle events
        running = renderer.handle_events()
        
        # Animate walls
        inner_radius = min(inner_radius + 0.002, outer_radius - 0.1)
        
        # Draw frame
        renderer.clear()
        renderer.draw_walls(inner_radius, outer_radius, 1, 0)
        renderer.draw_cities(positions, size=6.0)
        renderer.swap_buffers()
        
        # Limit framerate
        renderer.limit_framerate(60)
        
        frame += 1
    
    renderer.close()
    print("Animation test complete")