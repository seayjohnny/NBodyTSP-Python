"""
OpenGL Renderer for N-Body TSP Simulator

Provides real-time GPU-accelerated visualization using OpenGL.
"""

import numpy as np
from typing import Optional, Tuple
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

class TSPRenderer:
    """OpenGL renderer for visualizing N-body TSP simulation."""
    
    def __init__(self, window_size: Tuple[int, int] = (800, 800), 
                 title: str = "N-Body TSP Simulator"):
        """
        Initialize the renderer.
        
        Args:
            window_size: (width, height) of window
            title: Window title
        """
        self.window_size = window_size
        self.title = title
        self.window = None
        self.clock = None
        self.is_initialized = False
        
        # Colors (RGB)
        self.color_background = (0.0, 0.0, 0.0)
        self.color_city = (1.0, 0.0, 0.0)
        self.color_path = (0.0, 1.0, 0.0)
        self.color_wall_contract = (1.0, 0.0, 0.0)
        self.color_wall_static = (1.0, 1.0, 0.0)
        self.color_wall_expand = (0.0, 0.0, 1.0)
        self.color_bubble = (0.2, 0.8, 1.0)
        self.color_density = (0.5, 0.5, 0.5)
        
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
            
            # Set up 2D orthographic projection
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            
            # Enable features
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_POINT_SMOOTH)
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
            
            # Set background color
            glClearColor(*self.color_background, 1.0)
            
            # Focus window on Windows
            if WIN_GUI_AVAILABLE:
                self.focus_window()

            self.is_initialized = True
            print(f"Renderer initialized: {self.window_size[0]}x{self.window_size[1]}")
            return True
            
        except Exception as e:
            print(f"Failed to initialize renderer: {e}")
            return False
    
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
                  inner_direction: int, outer_direction: int):
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
            self.draw_circle((0, 0), inner_radius, color=inner_color, width=2.0)
        
        # Draw outer wall
        outer_color = get_color(outer_direction)
        self.draw_circle((0, 0), outer_radius, color=outer_color, width=2.0)
    
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
                # Draw with transparency
                glColor4f(*self.color_bubble, alpha)
                self.draw_circle((x, y), r, segments=50, filled=True)
                # Draw outline
                glColor3f(*self.color_bubble)
                self.draw_circle((x, y), r, segments=50, width=1.0)
    
    def draw_density_grid(self, density: np.ndarray, centers: np.ndarray,
                         bounds: Tuple[float, float, float, float],
                         bins: int, max_density: Optional[float] = None):
        """
        Draw density heatmap as colored rectangles.
        
        Args:
            density: Density values (bins*bins,)
            centers: Grid cell centers (bins*bins, 2)
            bounds: (min_x, min_y, max_x, max_y)
            bins: Number of bins per dimension
            max_density: Maximum density for color scaling
        """
        if not self.is_initialized:
            return
        
        if max_density is None:
            max_density = np.max(density) if len(density) > 0 else 1.0
        
        min_x, min_y, max_x, max_y = bounds
        cell_width = (max_x - min_x) / bins
        cell_height = (max_y - min_y) / bins
        
        for i in range(bins):
            for j in range(bins):
                idx = i + j * bins
                if idx >= len(density):
                    continue
                
                d = density[idx]
                if d > 0:
                    # Color intensity based on density
                    intensity = min(d / max_density, 1.0)
                    glColor4f(intensity, intensity, intensity, 0.3)
                    
                    # Draw rectangle
                    x = min_x + i * cell_width
                    y = min_y + j * cell_height
                    
                    glBegin(GL_QUADS)
                    glVertex2f(x, y)
                    glVertex2f(x + cell_width, y)
                    glVertex2f(x + cell_width, y + cell_height)
                    glVertex2f(x, y + cell_height)
                    glEnd()
    
    def draw_frame_complete(self, positions: np.ndarray, inner_radius: float,
                           outer_radius: float, inner_dir: int, outer_dir: int,
                           bubbles: Optional[np.ndarray] = None,
                           path: Optional[np.ndarray] = None,
                           coords: Optional[np.ndarray] = None):
        """
        Draw a complete frame with all elements.
        
        Args:
            positions: Current city positions
            inner_radius: Inner wall radius
            outer_radius: Outer wall radius
            inner_dir: Inner wall direction
            outer_dir: Outer wall direction
            bubbles: Optional bubble array
            path: Optional path to draw
            coords: Original coordinates (needed if drawing path)
        """
        self.clear()
        
        # Draw bubbles (background)
        if bubbles is not None:
            self.draw_bubbles(bubbles)
        
        # Draw walls
        self.draw_walls(inner_radius, outer_radius, inner_dir, outer_dir)
        
        # Draw path if provided
        if path is not None and coords is not None:
            self.draw_path(coords, path, width=2.0, color=(0.0, 1.0, 0.0))
        
        # Draw cities (foreground)
        self.draw_cities(positions, size=5.0)
        
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
        
        return True
    
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