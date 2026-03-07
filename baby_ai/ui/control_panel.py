import threading
import tkinter as tk
import ctypes
from typing import Callable

class AIControlPanel:
    def __init__(self, on_stop: Callable[[], None] = None):
        self.is_paused = False
        self.is_stopped = False
        self.root = None
        self.on_stop = on_stop

    def _run_ui(self):
        self.root = tk.Tk()
        self.root.title("Baby-AI Control")
        # Keep window always on top
        self.root.attributes("-topmost", True)
        
        # Position near the top right of the screen
        screen_width = self.root.winfo_screenwidth()
        x_pos = screen_width - 250
        y_pos = 50
        self.root.geometry(f"200x80+{x_pos}+{y_pos}")
        self.root.resizable(False, False)
        self.root.configure(padx=10, pady=10)
        
        self.btn_pause = tk.Button(
            self.root, 
            text="Pause AI", 
            command=self.toggle_pause,
            bg="#f0f0f0",
            font=("Arial", 10, "bold")
        )
        self.btn_pause.pack(fill=tk.X, pady=(0, 5))
        
        self.btn_stop = tk.Button(
            self.root, 
            text="Stop && Save", 
            command=self.trigger_stop,
            bg="#ffcccc",
            font=("Arial", 10, "bold")
        )
        self.btn_stop.pack(fill=tk.X)
        
        self.root.protocol("WM_DELETE_WINDOW", self.trigger_stop)
        
        # Poll to release mouse just in case user clicks a button and mouse got trapped unexpectedly
        self.poll_release_cursor()
        
        self.root.mainloop()

    def poll_release_cursor(self):
        if self.root:
            if self.is_paused or self.is_stopped:
                try:
                    ctypes.windll.user32.ClipCursor(None)
                except Exception:
                    pass
            self.root.after(1000, self.poll_release_cursor)

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.config(text="▶ Resume AI", bg="#ccffcc")
            # Immediately free the cursor when paused
            try:
                ctypes.windll.user32.ClipCursor(None)
            except Exception:
                pass
        else:
            self.btn_pause.config(text="⏸ Pause AI", bg="#f0f0f0")

    def trigger_stop(self):
        self.is_stopped = True
        try:
            ctypes.windll.user32.ClipCursor(None)
        except Exception:
            pass
        if self.on_stop:
            self.on_stop()
        if self.root:
            self.root.quit()

    def start(self):
        thread = threading.Thread(target=self._run_ui, daemon=True)
        thread.start()
