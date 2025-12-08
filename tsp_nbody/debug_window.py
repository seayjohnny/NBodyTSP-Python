"""
Debug Window for N-Body TSP Simulator

Displays real-time debug information for all particles in a separate Tkinter window.
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from typing import Optional, Dict


class DebugWindow:
    """Separate Tkinter window for displaying particle debug information."""

    def __init__(self, title: str = "N-Body Debug Info", n_particles: int = 0):
        """
        Initialize the debug window.

        Args:
            title: Window title
            n_particles: Number of particles (for initial sizing)
        """
        self.title = title
        self.n_particles = n_particles
        self.window = None
        self.tree = None
        self.is_open = False

        self._create_window()

    def _create_window(self):
        """Create the Tkinter window and table widget."""
        # Create main window
        self.window = tk.Tk()
        self.window.title(self.title)
        self.window.geometry("900x600")

        # Handle window close event
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

        # Create frame for treeview and scrollbars
        frame = ttk.Frame(self.window)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create scrollbars
        vsb = ttk.Scrollbar(frame, orient=tk.VERTICAL)
        hsb = ttk.Scrollbar(frame, orient=tk.HORIZONTAL)

        # Create treeview (table)
        columns = ("Index", "X", "Y", "Vx", "Vy", "Ax", "Ay", "Distance")
        self.tree = ttk.Treeview(
            frame,
            columns=columns,
            show="headings",
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
        )

        # Configure scrollbars
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)

        # Define column headings and widths
        column_widths = {
            "Index": 60,
            "X": 100,
            "Y": 100,
            "Vx": 100,
            "Vy": 100,
            "Ax": 100,
            "Ay": 100,
            "Distance": 100,
        }

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=column_widths[col], anchor=tk.E)  # Right-align

        # Grid layout
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        # Configure grid weights for resizing
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        # Add status label
        self.status_label = ttk.Label(
            self.window, text="Waiting for data...", relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        self.is_open = True

        # Process initial events
        self.window.update_idletasks()

    def update(self, debug_data: Dict):
        """
        Update the debug window with new particle data.

        Args:
            debug_data: Dictionary containing:
                - 'positions': np.ndarray (n, 2)
                - 'velocities': np.ndarray (n, 2)
                - 'accelerations': np.ndarray (n, 2)
                - 'distances': np.ndarray (n,)
                - 'step': int (optional)
        """
        if not self.is_open or self.window is None:
            return

        try:
            # Extract data
            positions = debug_data.get("positions", np.array([]))
            velocities = debug_data.get("velocities", np.array([]))
            accelerations = debug_data.get("accelerations", np.array([]))
            distances = debug_data.get("distances", np.array([]))
            step = debug_data.get("step", 0)

            # Validate data
            if len(positions) == 0:
                return

            n = len(positions)

            # Clear existing data
            for item in self.tree.get_children():
                self.tree.delete(item)

            # Populate table with new data
            for i in range(n):
                x, y = positions[i]
                vx, vy = velocities[i] if i < len(velocities) else (0.0, 0.0)
                ax, ay = accelerations[i] if i < len(accelerations) else (0.0, 0.0)
                dist = distances[i] if i < len(distances) else 0.0

                # Format values to 4 decimal places
                values = (
                    i,
                    f"{x:.4f}",
                    f"{y:.4f}",
                    f"{vx:.4f}",
                    f"{vy:.4f}",
                    f"{ax:.4f}",
                    f"{ay:.4f}",
                    f"{dist:.4f}",
                )

                self.tree.insert("", tk.END, values=values)

            # Update status
            self.status_label.config(text=f"Step: {step} | Particles: {n}")

            # Process events without blocking
            self.window.update_idletasks()

        except Exception as e:
            print(f"Error updating debug window: {e}")

    def is_visible(self) -> bool:
        """Check if the debug window is still open."""
        return self.is_open and self.window is not None

    def _on_close(self):
        """Handle window close event."""
        self.is_open = False
        if self.window:
            self.window.destroy()
            self.window = None

    def close(self):
        """Close the debug window and clean up resources."""
        if self.window:
            self._on_close()
