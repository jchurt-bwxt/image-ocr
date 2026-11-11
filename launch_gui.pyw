"""Windowed launcher for desktop use, including readable startup errors."""
from tkinter import messagebox

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        from image_ocr_gui import launch_gui
        launch_gui()
    except Exception as exc:
        messagebox.showerror(
            "Image OCR could not start",
            f"Check that the dependencies in requirements.txt are installed in your Python environment.\n\n{exc}",
        )
