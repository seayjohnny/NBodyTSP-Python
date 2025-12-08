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
    color_background: ColorType
    color_city: ColorType
    color_path: ColorType
    color_wall_contract: ColorType
    color_wall_static: ColorType
    color_wall_expand: ColorType
    color_bubble: ColorType
    color_density: ColorType
    color_text: ColorType


default_renderer_options: RendererOptions = {
    "window_size": (800, 800),
    "title": "N-Body TSP Simulator",
    "city_size": 5.0,
    "path_width": 2.0,
    "wall_width": 2.0,
    "padding": 0.1,  # 10% padding around edges
    "color_background": (0.9, 0.9, 1.0),
    "color_city": (0.2, 0.0, 1.0),
    "color_path": (0.0, 0.5, 0.0),
    "color_wall_contract": (1.0, 0.0, 0.0),
    "color_wall_static": (0.3, 0.3, 0.0),
    "color_wall_expand": (0.0, 0.0, 1.0),
    "color_bubble": (0.2, 0.8, 1.0),
    "color_density": (0.5, 0.5, 0.5),
    "color_text": (1.0, 1.0, 1.0),
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

        # Sizes
        self.city_size = options.get("city_size", 5.0)  # Point size for cities
        self.path_width = options.get("path_width", 2.0)  # Line width for paths
        self.wall_width = options.get("wall_width", 2.0)  # Line width for walls
        self.padding = options.get("padding", 0.1)  # Padding fraction

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

    def clear(self):
        """Clear the display."""
        if not self.is_initialized:
            return
        glClear(GL_COLOR_BUFFER_BIT)
    
    def draw_cities(self, positions: np.ndarray, size: float = 5.0, 
                   color: Optional[Tuple[float, float, float]] = None):
        """
        Draw cities as points.
        
        Args:
            positions: City positions array (n, 2)
            size: Point size
            color: RGB color tuple, or None for default
        """
        if not self.is_initialized:
            return
        
        color = color or self.color_city
        
        glPointSize(size)
        glColor3f(*color)
        
        glBegin(GL_POINTS)
        for pos in positions:
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
                  inner_direction: int, outer_direction: int, width: float = 2.0):
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
            self.draw_circle((0, 0), inner_radius, color=inner_color, width=width)
        
        # Draw outer wall
        outer_color = get_color(outer_direction)
        self.draw_circle((0, 0), outer_radius, color=outer_color, width=width)
    
    def draw_path(self, coords: np.ndarray, path: np.ndarray,
                 width: float = 3.0, color: Optional[Tuple[float, float, float]] = None):
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
        
        glLineWidth(width)
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
        if not self.is_initialized or bubbles is None or len(bubbles) == 0:
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
                         bins: int, max_density: Optional[float] = None,
                         draw_grid_lines: bool = True):
        """
        Draw density heatmap as colored rectangles with grid lines.

        Args:
            density: Density values (bins*bins,), or None
            bins: Number of bins per dimension
            max_density: Maximum density for color scaling
            draw_grid_lines: Whether to draw grid lines
        """
        if not self.is_initialized:
            return

        min_x, min_y = -1.0, -1.0
        cell_width = 2.0 / bins
        cell_height = 2.0 / bins

        # Draw colored rectangles for density (if data provided)
        if density is not None and len(density) > 0:
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

                        # Use a color gradient: orange-red for density
                        # More intense = more particles
                        glColor4f(intensity, intensity * 0.5, 0.0, 0.4)

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
        if draw_grid_lines:
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
                if event.button == 1:  # Left mouse button
                    self.toggle_pause()

        return True

    def toggle_pause(self):
        """Toggle pause state and print status."""
        self.is_paused = not self.is_paused
        status = "PAUSED" if self.is_paused else "RESUMED"
        print(f"Simulation {status}")
    
    def wait_for_key(self):
        """Wait for user to press a key."""
        if not self.is_initialized:
            return
        
        print("Press any key to continue...")
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN:
                    waiting = False
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