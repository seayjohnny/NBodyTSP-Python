# ruff:noqa: F403 F405
"""
3D OpenGL Renderer for Torus N-Body TSP Simulator.

Replicates the WebGL renderer from the JavaScript NBodyTSP-Torus project:
- Perspective camera with mouse orbit/zoom
- Wireframe torus mesh that updates as tube collapses
- Point sprites with energy-based coloring
- Circle + tour route lines in circle phase
- City ID labels
"""

import numpy as np
import math
import sys
from typing import Optional, Tuple

TAU = 2.0 * np.pi

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


def _gen_torus_wireframe(R: float, r: float, seg_u: int = 48, seg_v: int = 24):
    """Generate wireframe torus vertices and line indices."""
    verts = []
    for i in range(seg_u + 1):
        u = (i / seg_u) * TAU
        cu, su = math.cos(u), math.sin(u)
        for j in range(seg_v + 1):
            v = (j / seg_v) * TAU
            cv, sv = math.cos(v), math.sin(v)
            verts.extend([
                (R + r * cv) * cu,
                (R + r * cv) * su,
                r * sv,
            ])

    indices = []
    # Meridian lines (along u for each v)
    for j in range(seg_v):
        for i in range(seg_u):
            a = i * (seg_v + 1) + j
            b = (i + 1) * (seg_v + 1) + j
            indices.extend([a, b])

    # Parallel lines (along v for each u)
    for i in range(seg_u + 1):
        for j in range(seg_v):
            a = i * (seg_v + 1) + j
            indices.extend([a, a + 1])

    return np.array(verts, dtype=np.float32), np.array(indices, dtype=np.uint32)


def _gen_circle(R: float, segs: int = 128):
    """Generate circle vertices on the xy plane."""
    verts = []
    for i in range(segs + 1):
        t = (i / segs) * TAU
        verts.extend([R * math.cos(t), R * math.sin(t), 0.0])
    return np.array(verts, dtype=np.float32)


class TorusRenderer:
    """3D OpenGL renderer replicating the JS torus visualization."""

    def __init__(
        self,
        sim_size: Tuple[int, int] = (800, 800),
        panel_width: int = 0,
        title: str = "N-Body TSP · Torus Collapse",
    ):
        self.sim_size = sim_size
        self.panel_width = panel_width
        self.title = title
        self.window = None
        self.clock = None
        self.is_initialized = False

        # Camera
        self.cam_angle = 0.4
        self.cam_pitch = 0.35
        self.cam_dist = 8.5
        self.is_dragging = False
        self.last_mx = 0
        self.last_my = 0
        self.auto_rotate = True

        # Rendering state
        self.show_labels = True
        self.font = None
        self.label_surface = None  # Pygame surface for text overlay

        # Control panel
        self.control_panel = None

        # Pause
        self.is_paused = False

    def initialize(self) -> bool:
        if not OPENGL_AVAILABLE:
            return False

        try:
            pygame.init()

            total_w = self.sim_size[0] + self.panel_width
            total_h = self.sim_size[1]
            self.window = pygame.display.set_mode(
                (total_w, total_h), DOUBLEBUF | OPENGL
            )
            pygame.display.set_caption(self.title)
            self.clock = pygame.time.Clock()

            # OpenGL setup
            sim_w, sim_h = self.sim_size
            glViewport(0, 0, sim_w, sim_h)

            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LEQUAL)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_POINT_SMOOTH)
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

            # Background: warm off-white like JS (#f4f1ec)
            glClearColor(0.957, 0.945, 0.925, 1.0)

            # Font for labels
            try:
                self.font = pygame.font.SysFont("consolas", 12)
            except Exception:
                self.font = pygame.font.Font(None, 14)

            # Label overlay surface (transparent)
            self.label_surface = pygame.Surface(
                (sim_w, sim_h), pygame.SRCALPHA
            )

            if WIN_GUI_AVAILABLE:
                try:
                    hwnd = pygame.display.get_wm_info()['window']
                    win32gui.SetForegroundWindow(hwnd)
                except Exception:
                    pass

            self.is_initialized = True
            print(f"Torus renderer initialized: {sim_w}x{sim_h}" +
                  (f" + {self.panel_width}px panel" if self.panel_width > 0 else ""))
            return True

        except Exception as e:
            print(f"Failed to initialize torus renderer: {e}")
            return False

    def _setup_perspective(self):
        """Set up perspective projection and camera view."""
        sim_w, sim_h = self.sim_size
        glViewport(0, 0, sim_w, sim_h)

        aspect = sim_w / sim_h
        fov = 45.0

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(fov, aspect, 0.1, 100.0)

        # Camera position from spherical coordinates
        eye_x = self.cam_dist * math.cos(self.cam_pitch) * math.cos(self.cam_angle)
        eye_y = self.cam_dist * math.cos(self.cam_pitch) * math.sin(self.cam_angle)
        eye_z = self.cam_dist * math.sin(self.cam_pitch)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            eye_x, eye_y, eye_z,  # eye
            0, 0, 0,               # center
            0, 0, 1,               # up
        )

    def _get_vp_matrix(self):
        """Get the current view-projection matrix for label projection."""
        proj = glGetDoublev(GL_PROJECTION_MATRIX)
        model = glGetDoublev(GL_MODELVIEW_MATRIX)
        # Multiply model * proj as numpy
        return np.array(model) @ np.array(proj)

    def _project_3d_to_screen(self, pos, vp_matrix):
        """Project a 3D point to screen coordinates."""
        sim_w, sim_h = self.sim_size
        x, y, z = pos[0], pos[1], pos[2]

        # Homogeneous coords
        clip = np.array([
            vp_matrix[0, 0] * x + vp_matrix[1, 0] * y + vp_matrix[2, 0] * z + vp_matrix[3, 0],
            vp_matrix[0, 1] * x + vp_matrix[1, 1] * y + vp_matrix[2, 1] * z + vp_matrix[3, 1],
            vp_matrix[0, 2] * x + vp_matrix[1, 2] * y + vp_matrix[2, 2] * z + vp_matrix[3, 2],
            vp_matrix[0, 3] * x + vp_matrix[1, 3] * y + vp_matrix[2, 3] * z + vp_matrix[3, 3],
        ])

        if clip[3] < 0.01:
            return None

        ndc_x = clip[0] / clip[3]
        ndc_y = clip[1] / clip[3]

        screen_x = (ndc_x * 0.5 + 0.5) * sim_w
        screen_y = (1.0 - (ndc_y * 0.5 + 0.5)) * sim_h

        return (int(screen_x), int(screen_y))

    def render(
        self,
        particles_pos: np.ndarray,
        particles_vel: np.ndarray,
        R: float,
        r: float,
        r0: float,
        circle_phase: bool,
        found_tour: Optional[list] = None,
        particle_ids: Optional[list] = None,
    ):
        """
        Render a complete frame.

        Args:
            particles_pos: (N, 3) positions
            particles_vel: (N, 3) velocities
            R: Major circle radius
            r: Current tube radius
            r0: Initial tube radius (for alpha calculation)
            circle_phase: Whether in circle phase
            found_tour: Optional 1-indexed tour order
            particle_ids: Optional list of particle IDs (1-indexed)
        """
        if not self.is_initialized:
            return

        # Auto-rotate camera
        if self.auto_rotate and not self.is_dragging:
            self.cam_angle += 0.0015

        # Clear
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set up 3D perspective
        self._setup_perspective()

        n = len(particles_pos)

        # --- Draw particles as points ---
        glEnable(GL_BLEND)
        glDepthMask(GL_TRUE)

        point_size = 12.0 if circle_phase else 8.0
        glPointSize(point_size)

        glBegin(GL_POINTS)
        for i in range(n):
            # Energy-based coloring (like JS)
            speed = np.sqrt(np.sum(particles_vel[i] ** 2))
            energy = min(1.0, speed * 2.5)

            # Color gradient: core(orange-brown) -> hot(bright orange) via energy
            core = np.array([0.78, 0.24, 0.08])
            hot = np.array([0.96, 0.52, 0.18])
            color = core + (hot - core) * energy

            glColor4f(color[0], color[1], color[2], 0.9)
            glVertex3f(particles_pos[i, 0], particles_pos[i, 1], particles_pos[i, 2])
        glEnd()

        # --- Draw torus wireframe (only in torus phase) ---
        if not circle_phase:
            torus_verts, torus_indices = _gen_torus_wireframe(R, r, 48, 24)

            alpha = max(0.04, (r / max(r0, 0.01)) * 0.28)
            glDepthMask(GL_FALSE)
            glColor4f(0.38, 0.30, 0.24, alpha)
            glLineWidth(1.0)

            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, torus_verts)
            glDrawElements(GL_LINES, len(torus_indices), GL_UNSIGNED_INT, torus_indices)
            glDisableClientState(GL_VERTEX_ARRAY)
            glDepthMask(GL_TRUE)

        # --- Circle phase rendering ---
        if circle_phase:
            # Draw the major circle
            circle_verts = _gen_circle(R, 128)
            glColor4f(0.4, 0.3, 0.2, 0.15)
            glLineWidth(1.5)
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, circle_verts)
            glDrawArrays(GL_LINE_STRIP, 0, 129)
            glDisableClientState(GL_VERTEX_ARRAY)

            # Draw found tour route
            if found_tour is not None and particle_ids is not None:
                id_to_pos = {}
                id_to_theta = {}
                for i in range(n):
                    pid = particle_ids[i] if i < len(particle_ids) else i + 1
                    id_to_pos[pid] = particles_pos[i]
                    id_to_theta[pid] = math.atan2(particles_pos[i, 1], particles_pos[i, 0])

                # Found tour (orange, slightly inside circle)
                glColor4f(0.75, 0.22, 0.08, 0.65)
                glLineWidth(2.0)
                glBegin(GL_LINE_STRIP)
                for k in range(len(found_tour) + 1):
                    pid = found_tour[k % len(found_tour)]
                    if pid in id_to_theta:
                        th = id_to_theta[pid]
                        rr = R - 0.12
                        glVertex3f(rr * math.cos(th), rr * math.sin(th), 0.0)
                glEnd()

        # --- City ID labels ---
        if self.show_labels and self.font and particle_ids is not None:
            # Get VP matrix for projection
            proj_mat = np.array(glGetDoublev(GL_PROJECTION_MATRIX))
            model_mat = np.array(glGetDoublev(GL_MODELVIEW_MATRIX))
            mvp = model_mat @ proj_mat

            self.label_surface.fill((0, 0, 0, 0))

            for i in range(n):
                pid = particle_ids[i] if i < len(particle_ids) else i + 1
                screen = self._project_3d_to_screen(particles_pos[i], mvp)
                if screen is not None:
                    label = self.font.render(str(pid), True, (42, 37, 32, 153))
                    lw = label.get_width()
                    self.label_surface.blit(label, (screen[0] - lw // 2, screen[1] - 16))

            # Blit label surface over OpenGL
            self._blit_surface_over_gl(self.label_surface)

        # --- Draw control panel ---
        if self.control_panel and self.panel_width > 0:
            self._draw_panel()

        pygame.display.flip()

    def _surface_to_texture(self, surface: pygame.Surface) -> int:
        """Upload a pygame Surface to an OpenGL texture, return texture ID."""
        tex_data = pygame.image.tostring(surface, "RGBA", True)
        w, h = surface.get_size()

        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, tex_data)
        return tex_id

    def _draw_textured_quad(self, x, y, w, h):
        """Draw a textured quad at screen position (x,y) with size (w,h)."""
        glEnable(GL_TEXTURE_2D)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(x, y + h)
        glTexCoord2f(1, 0); glVertex2f(x + w, y + h)
        glTexCoord2f(1, 1); glVertex2f(x + w, y)
        glTexCoord2f(0, 1); glVertex2f(x, y)
        glEnd()
        glDisable(GL_TEXTURE_2D)

    def _blit_surface_over_gl(self, surface: pygame.Surface):
        """Blit a pygame surface on top of the OpenGL framebuffer using a textured quad."""
        sim_w, sim_h = self.sim_size
        sw, sh = surface.get_size()

        tex_id = self._surface_to_texture(surface)

        # Switch to 2D ortho for overlay
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, sim_w, sim_h, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(1, 1, 1, 1)

        glBindTexture(GL_TEXTURE_2D, tex_id)
        self._draw_textured_quad(0, 0, sw, sh)

        glEnable(GL_DEPTH_TEST)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        glDeleteTextures([tex_id])

    def _draw_panel(self):
        """Draw the UI control panel on the right side using a textured quad."""
        if not self.control_panel:
            return

        panel_surface = self.control_panel.draw()
        sim_w, sim_h = self.sim_size
        pw, ph = panel_surface.get_size()

        tex_id = self._surface_to_texture(panel_surface)

        # Set viewport to panel area
        glViewport(sim_w, 0, self.panel_width, sim_h)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, pw, ph, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(1, 1, 1, 1)

        glBindTexture(GL_TEXTURE_2D, tex_id)
        self._draw_textured_quad(0, 0, pw, ph)

        glEnable(GL_DEPTH_TEST)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        glDeleteTextures([tex_id])

        # Restore sim viewport
        glViewport(0, 0, sim_w, sim_h)

    def handle_events(self) -> bool:
        """
        Process window events. Returns False if quit requested.
        Also returns dict of panel changes via self.last_panel_changes.
        """
        self.last_panel_changes = {}

        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return False
                if event.key == K_SPACE:
                    self.is_paused = not self.is_paused
                if event.key == K_l:
                    self.show_labels = not self.show_labels

            # Camera controls (only in sim area)
            if event.type == MOUSEBUTTONDOWN:
                mx, my = event.pos
                if mx < self.sim_size[0]:  # In sim area
                    if event.button == 1:
                        self.is_dragging = True
                        self.last_mx = mx
                        self.last_my = my
                    elif event.button == 4:  # Scroll up
                        self.cam_dist = max(3.0, self.cam_dist - 0.3)
                    elif event.button == 5:  # Scroll down
                        self.cam_dist = min(18.0, self.cam_dist + 0.3)

            if event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    self.is_dragging = False

            if event.type == MOUSEMOTION and self.is_dragging:
                mx, my = event.pos
                dx = mx - self.last_mx
                dy = my - self.last_my
                self.cam_angle += dx * 0.005
                self.cam_pitch += dy * 0.005
                self.cam_pitch = max(-1.2, min(1.2, self.cam_pitch))
                self.last_mx = mx
                self.last_my = my

            if event.type == MOUSEWHEEL:
                self.cam_dist = max(3.0, min(18.0, self.cam_dist - event.y * 0.5))

            # Panel events
            if self.control_panel and self.panel_width > 0:
                changes = self.control_panel.handle_event(event, self.sim_size[0])
                self.last_panel_changes.update(changes)

        return True

    def close(self):
        if self.is_initialized:
            pygame.quit()
            self.is_initialized = False
            print("Torus renderer closed")
